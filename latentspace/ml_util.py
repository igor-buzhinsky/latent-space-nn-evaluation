import os
from typing import *
import time
import random
from enum import Enum
from abc import ABC, abstractmethod
import glob

import numpy as np
import scipy
import scipy.stats
import statsmodels.nonparametric.bandwidths
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2

if os.name == "posix":
    import resource

# A enumeration of all supported datasets.
DatasetInfo = Enum("DatasetInfo", "MNIST CelebA128Gender LSUN128 ImageNet")

### PyTorch utils ###

class Lambda(torch.nn.Module):
    def __init__(self, forward):
        super().__init__()
        self.lambda_forward = forward

    def forward(self, x):
        return self.lambda_forward(x)


class NopLayer(torch.nn.Module):
    def forward(self, x):
        return x
    

class Flatten(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view(x.shape[0], -1)
    

class Upsample(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.nn.functional.interpolate(x, size=None, scale_factor=2, mode='bilinear', align_corners=False)


class Resize(torch.nn.Module):
    def __init__(self, side: int):
        super().__init__()
        self.side = side
    
    def forward(self, x: torch.Tensor):
        return torch.nn.functional.interpolate(x, size=(self.side, self.side),
                                               mode="bicubic", align_corners=False)
    
    
class Adversary(ABC):
    """
    Base class for adversaries. Adversaries can perturb vectors given the gradient pointing to the direction
    of making the prediction worse.
    """
    
    @abstractmethod
    def perturb(self, initial_vector: torch.Tensor,
                get_gradient: Callable[[torch.Tensor], Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """
        Perturb the given vector. 
        :param initial_vector: initial vector. If this is the original image representation, it must be flattened
            prior to the call (more precisely, it must be of size [1, the_rest]).
        :param get_gradient: a get_gradient function. It accepts the current vector and returns a tuple
            (gradient pointing to the direction of the adversarial attack, the corresponding loss function value).
        :return: the pertured vector of the same size as initial_vector.
        """
        pass

    
class NopAdversary(Adversary):
    """
    Dummy adversary that acts like an identity function.
    """
    
    def perturb(self, initial_vector: torch.Tensor,
                get_gradient: Callable[[torch.Tensor], Tuple[torch.Tensor, float]]) -> torch.Tensor:
        return initial_vector


class PGDAdversary(Adversary):
    """
    Performes Projected Gradient Descent (PGD), or, more precisely, PG ascent according to the provided gradient.
    """
    
    def __init__(self, rho: float = 0.1, steps: int = 25, step_size: float = 0.1, random_start: bool = True,
                 stop_loss: float = 0, verbose: int = 1, norm: str = "scaled_l_2",
                 n_repeat: int = 1, repeat_mode: str = None, unit_sphere_normalization: bool = False):
        """
        Constrauts PGDAdversary. 
        :param rho > 0: bound on perturbation norm.
        :param steps: number of steps to perform in each run. Less steps can be done if stop_loss is reached.
        :param step_size: step size. Each step will be of magnitude rho * step_size.
        :param random_start: if True, start search in a vector with a uniformly random radius within the rho-ball.
            Otherwise, start in the center of the rho-ball.
        :param stop_loss: the search will stop when this value of the "loss" function is exceeded.
        :param verbose: 0 (silent), 1 (regular), 2 (verbose).
        :param norm: one of 'scaled_l_2' (default), 'l_2' or 'l_inf'.
        :param n_repeat: number of times to run PGD.
        :param repeat_mode: 'any' or 'min': In mode 'any', n_repeat runs are identical and any run that reaches
            stop_loss will prevent subsequent runs. In mode 'min', all runs will be performed, and if a run
            finds a smaller perturbation according to norm, it will tighten rho on the next run.
        :param unit_sphere_normalization: search perturbations on the unit sphere (according to the scaled L2 norm)
            instead of the entire latent space.
        """
        super().__init__()
        self.rho = rho
        self.steps = steps
        self.step_size = step_size
        self.random_start = random_start
        self.stop_loss = stop_loss
        self.verbose = verbose
        # checks on norms
        assert norm in ["scaled_l_2", "l_2", "l_inf"], "norm must be either 'scaled_l_2', 'l_2' or 'l_inf'"
        self.scale_norm = norm == "scaled_l_2"
        self.inf_norm = norm == "l_inf"
        # checks on repeated runs
        assert n_repeat >= 1, "n_repeat must be positive"
        assert not(n_repeat > 1 and repeat_mode is None), "if n_repeat > 1, repeat_mode must be set" 
        assert repeat_mode in [None, "any", "min"], "if repeat_mode is set, it must be either 'any' or 'min'"
        self.n_repeat = n_repeat
        self.shrinking_repeats = repeat_mode == "min"
        # optional unit sphere normalization
        self.unit_sphere_normalization = unit_sphere_normalization
        assert not unit_sphere_normalization or norm == "scaled_l_2",\
            "unit_sphere_normalization is only compatible with scaled_l_2 norm"
    
    def _norm(self, x: torch.Tensor) -> float:
        """
        (Possibly scaled) norm of x.
        """
        return x.norm(np.infty if self.inf_norm else 2).item() / (np.sqrt(x.numel()) if self.scale_norm else 1)
    
    def _normalize_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the vector of gradients.
        In the L2 space, this is done by dividing the vector by its norm.
        In the L-inf space, this is done by taking the sign of the gradient.
        """
        return x.sign() if self.inf_norm else (x / self._norm(x))
    
    def _project(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        """
        Projects the vector onto the rho-ball.
        In the L2 space, this is done by scaling the vector.
        In the L-inf space, this is done by clamping all components independently.
        """
        return x.clamp(-rho, rho) if self.inf_norm else (x / self._norm(x) * rho)
    
    def perturb(self, initial_vector: torch.Tensor,
                get_gradient: Callable[[torch.Tensor], Tuple[torch.Tensor, float]]) -> torch.Tensor:
        best_perturbation = None
        best_perturbation_norm = np.infty
        # rho may potentially shrink with repeat_mode == "min":
        rho = self.rho
        random_start = self.random_start
        for run_n in range(self.n_repeat):
            x1 = initial_vector * 1
            perturbation = x1 * 0

            if random_start:
                # random vector within the rho-ball
                if self.inf_norm:
                    # uniform
                    perturbation = (torch.rand(1, x1.numel()) - 0.5) * 2 * rho
                    # possibly reduce radius to encourage search of vectors with smaller norms
                    perturbation *= np.random.rand()
                else:
                    # uniform radius, random direction
                    # note that this distribution is not uniform in terms of R^n!
                    perturbation = torch.randn(1, x1.numel())
                    if rho > 0:
                        perturbation /= self._norm(perturbation) / rho
                    else:
                        perturbation *= 0
                    perturbation *= np.random.rand()
                perturbation = Util.conditional_to_cuda(perturbation)

            if self.verbose > 0:
                print(f">> #run = {run_n}, ║x1║ = {self._norm(x1):.5f}, ρ = {rho:.5f}")

            found = False
            for i in range(self.steps):
                perturbed_vector = x1 + perturbation
                perturbed_vector, perturbation = self._recompute_with_unit_sphere_normalization(perturbed_vector,
                                                                                                perturbation)
                classification_gradient, classification_loss = get_gradient(perturbed_vector)
                if self.verbose > 0:
                    if classification_loss > self.stop_loss or i == self.steps - 1 or i % 5 == 0 and self.verbose > 1:
                        print(f"step {i:3d}: objective = {-classification_loss:+.7f}, "
                              f"║Δx║ = {self._norm(perturbation):.5f}, ║x║ = {self._norm(perturbed_vector):.5f}")
                if classification_loss > self.stop_loss:
                    found = True
                    break
                # learning step
                perturbation_step = rho * self.step_size * self._normalize_gradient(classification_gradient)
                perturbation_step = self._adjust_step_for_unit_sphere(perturbation_step, x1 + perturbation)
                perturbation += perturbation_step
                # projecting on rho-ball around x1
                if self._norm(perturbation) > rho:
                    perturbation = self._project(perturbation, rho)
            
            # end of run
            if found:
                if self.shrinking_repeats:
                    if self._norm(perturbation) < best_perturbation_norm:
                        best_perturbation_norm = self._norm(perturbation)
                        best_perturbation = perturbation
                        rho = best_perturbation_norm
                else: # regular repeats
                    # return immediately
                    return self._optional_normalize(x1 + perturbation)
            if best_perturbation is None:
                best_perturbation = perturbation
            if self.shrinking_repeats and run_n == self.n_repeat - 1:
                # heuristic: the last run is always from the center
                random_start = False
        return self._optional_normalize(x1 + best_perturbation)
    
    ### projections on the unit sphere: ###
    
    def _optional_normalize(self, x: torch.Tensor):
        """
        Optional unit sphere normalization.
        :param x: vector of shape 1*dim to normalize.
        :return optionally normalized x.
        """
        return Util.normalize_latent(x) if self.unit_sphere_normalization else x
    
    def _recompute_with_unit_sphere_normalization(self, perturbed_vector: torch.Tensor, perturbation: torch.Tensor):
        """
        If unit sphere normalization is enabled, the perturbed vector is projected on the unit sphere,
        and the perturbation vector is recomputed accordingly. Otherwise, returns the inputs unmodified.
        :param perturbed_vector: perturbed vector (initial vector + perturbation).
        :param perturbation: perturbation vector.
        :return possibly recomputed (perturbed_vector, perturbation).
        """
        effective_perturbed_vector = self._optional_normalize(perturbed_vector)
        return effective_perturbed_vector, perturbation + effective_perturbed_vector - perturbed_vector
    
    def _adjust_step_for_unit_sphere(self, perturbation_step: torch.Tensor, previous_perturbed_vector: torch.Tensor):
        """
        If unit sphere normalization is enabled, multiplies perturbation_step by a coefficient that approximately
        compensates for the reduction of the learning step due to projection of a unit sphere.
        :param perturbation_step: unmodified pertubation step.
        :param previous_perturbed_vector: previous perturbed vector.
        :return altered perturbation_step.
        """
        new_perturbed_vector = self._optional_normalize(previous_perturbed_vector + perturbation_step)
        effective_perturbation_step = new_perturbed_vector - previous_perturbed_vector
        coef = perturbation_step.norm() / effective_perturbation_step.norm()
        return perturbation_step * coef
        

class ImageSet:
    """
    Accumulates images and captions. Shows them in blocks of size `max_num'.
    """
    
    def __init__(self, max_num: int):
        """
        Constructs ImageSet.
        :param max_num: number of images and captions to accumulate until they can be shown.
        """
        self.images = []
        self.captions = []
        self.max_num = max_num
    
    def append(self, images: List[torch.Tensor], captions: List[str] = None):
        """
        Appends the given images and captions.
        :param images: list of images (PyTorch tensors).
        :param captions: list of string captions corresponding to the images.
        """
        self.images += [images]
        if captions is None:
            captions = [""] * len(images)
        self.captions += [captions]
    
    def maybe_show(self, force: bool = False, nrow: int = 8):
        """
        Shows the images and their captions if a sufficient number of them has been accumulated.
        If the images and captions are shown, they are removed from the memory of ImageSet.
        :param force: if True, shows everything anyway. This may be useful to make the last call of maybe_show.
        :param nrow: nrow passed to Util.imshow_tensors.
        """
        if self.images and (force or len(self.images) >= self.max_num):
            Util.imshow_tensors(*sum(self.images, []), captions=sum(self.captions, []), nrow=nrow)
            self.images.clear()
            self.captions.clear()

            
class Util:
    """
    A convenience static class for everything that does not have its own class.
    """
    
    # if False, will use CPU even if CUDA is available
    cuda_enabled = True
    using_cuda = cuda_enabled and torch.cuda.is_available()
    
    @staticmethod
    def set_memory_limit(mb: int):
        """
        Sets memory limit in megabytes (this has effect only on Linux).
        """
        if os.name == "posix":
            rsrc = resource.RLIMIT_DATA
            soft, hard = resource.getrlimit(rsrc)
            resource.setrlimit(rsrc, (1024 * 1024 * mb, hard))
    
    @staticmethod
    def configure(memory_limit_mb: int):
        """
        Sets memory limit in megabytes (this has effect only on Linux). Configures matplotlib fonts.
        """
        Util.set_memory_limit(memory_limit_mb)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
    
    @staticmethod
    def tensor2numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()
    
    @staticmethod
    def imshow(img: torch.Tensor, figsize: Tuple[float, float] = (12, 2)):
        """
        Shows the image and saves the produced figure.
        :param img: image to show (PyTorch tensor).
        :param figsize: matplotlib figsize.
        """
        plt.figure(figsize=figsize)
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(Util.tensor2numpy(img), (1, 2, 0)))
        plt.axis("off")
        LogUtil.savefig("imshow")
        plt.show()
        plt.close()
    
    @staticmethod
    def imshow_tensors(*tensors: torch.Tensor, clamp: bool = True, figsize: Tuple[float, float] = (12, 2),
                       captions: List[str] = None, nrow: int = 8, pad_value: float = 1):       
        """
        Enhanced torchvision.utils.make_grid(). Supports image captions. Also saves the produced figure.
        :param tensors: images to show.
        :param clamp: of True, clamp all pixels to [-1, 1].
        :param figsize: matplotlib figsize.
        :param captions: list of string captions to be printed on top of images.
        :param nrow: nrow to be passed to torchvision.utils.make_grid.
        :param pad_value: pad_value to be passed to torchvision.utils.make_grid.
        """
        t = torch.cat(tensors)
        assert len(t.shape) == 4, f"Invalid shape of tensors {t.shape}"
        # handling 1D images
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        if clamp:
            t = torch.clamp(t, -1, 1)
        t = list(t)
            
        # adding textual captions if they are given (this involes rescaling)        
        if captions is not None:
            def multiline_puttext(img, caption: str):
                """
                cv2.putText does not support line breaks on its own.
                """
                scale = 0.8
                y0 = img.shape[0] * 0.15
                dy = img.shape[0] * 0.20
                for i, text in enumerate(caption.split("\n")):
                    y = int(y0 + i * dy)
                    # green
                    img = cv2.putText(img, text, (0, y), cv2.FONT_HERSHEY_TRIPLEX, scale, (0, 250, 0))
                return img
            default_size = (128,) * 2
            for i in range(len(t)):
                assert type(captions[i]) == str, "Captions must be str"
                t[i] = (Util.tensor2numpy(t[i]).transpose(1, 2, 0) / 2 + 0.5) * 255
                # shape = H*W*3
                t[i] = cv2.resize(t[i], default_size, interpolation=cv2.INTER_NEAREST)
                t[i] = multiline_puttext(t[i], captions[i]) / 255
                t[i] = torch.FloatTensor(t[i].transpose(2, 0, 1) * 2 - 1)
            
        Util.imshow(torchvision.utils.make_grid(torch.stack(t), nrow=nrow, pad_value=pad_value), figsize)
        
    @staticmethod
    def class_specific_loader(target_label: int, parent_loader_fn: Callable) -> Callable:
        """
        Filters the supplied loader parent_loader_fn, retaining only elements with target_label.
        Preserves batch size.
        :param target_label: the label to retain.
        :param parent_loader_fn: the loader-returning function to decorate.
        :return: decorated parent_loader_fn.
        """
        def loader():
            data_generator = iter(parent_loader_fn())
            result = []
            while True:
                try:
                    items, labels = next(data_generator)
                    batch_size = len(items)
                    # accumulate only items of target_label
                    result += [item for item, label in zip(items, labels) if label == target_label]
                    if len(result) >= batch_size:
                        yield torch.stack(result[:batch_size]), [target_label] * batch_size
                        # save the remainder for the next yield
                        result = result[batch_size:]
                except StopIteration:
                    return
        return loader
    
    @staticmethod
    def fixed_length_loader(no_images: int, parent_loader_fn: Callable, restarts: bool = True) -> Callable:
        """
        Restarts or limits the parent loader so that the desired number of images is produced.
        :param no_images: desired number of images (the effective number will be a multiple of batch size).
        :param parent_loader_fn: the loader-returning function to decorate.
        :param restarts: if False, then just limit the parent loader and do not restart it when the end is reached.
        :return: decorated parent_loader_fn.
        """
        def loader():
            generated = 0
            while generated < no_images:
                data_generator = iter(parent_loader_fn())
                while generated < no_images:
                    try:
                        items, labels = next(data_generator)
                    except StopIteration:
                        if restarts:
                            data_generator = iter(parent_loader_fn())
                            items, labels = next(data_generator)
                        else:
                            return
                    yield items, labels
                    generated += len(items)
        return loader
    
    @staticmethod
    def leakless_cycle(iterable_fn: Callable) -> Generator:
        """
        Fixes the memory leak problem of itertools.cycle (https://github.com/pytorch/pytorch/issues/23900).
        :iterable_fn function that returns an iterable.
        """
        iterator = iter(iterable_fn())
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable_fn())
    
    @staticmethod
    def optimizable_clone(x: torch.Tensor) -> torch.Tensor:
        """
        Clones a PyTorch tensor and makes it suitable for optimization.
        :param x: input tensor.
        :return: x with enabled gradients.
        """
        return Util.conditional_to_cuda(x.clone().detach()).requires_grad_(True)
    
    @staticmethod
    def set_param_requires_grad(m: torch.nn.Module, value: bool):
        """
        Sets requires_grad_(value) for all parameters of the module.
        :param m: PyTorch module.
        :param value: value to set.
        """
        for p in m.parameters():
            p.requires_grad_(value)
    
    @staticmethod
    def conditional_to_cuda(x: Union[torch.Tensor, torch.nn.Module]) -> torch.Tensor:
        """
        Returns the tensor/module on GPU if there is at least 1 GPU, otherwise just returns the tensor.
        :param x: a PyTorch tensor or module.
        :return: x on GPU if there is at least 1 GPU, otherwise just x.
        """
        return x.cuda() if Util.using_cuda else x
    
    @staticmethod
    def number_of_trainable_parameters(model: torch.nn.Module) -> int:
        """
        Number of trainable parameters in a PyTorch module, including nested modules.
        :param model: PyTorch module.
        :return: number of trainable parameters in model.
        """
        return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    
    @staticmethod
    def set_random_seed(seed: int = None):
        """
        Set random seed of random, numpy and pytorch.
        :param seed seed value. If None, it is replaced with the current timestamp.
        """
        if seed is None:
            seed = int(time.time())
        else:
            assert seed >= 0
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed_all(seed + 2)
        
    @staticmethod
    def normalize_latent(x: torch.Tensor) -> torch.Tensor:
        """
        Divides each latent vector of a batch by its scaled Euclidean norm.
        :param x: batch of latent vectors.
        :return normalized vector.
        """
        norm_vector = (np.sqrt(x.shape[1]) / torch.norm(x, dim=1)).unsqueeze(0)
        norm_vector = norm_vector.expand(x.shape[0], norm_vector.shape[1])
        return norm_vector @ x
        #return torch.stack([x[i] / torch.norm(x[i]) for i in range(x.shape[0])])
        
    @staticmethod
    def get_kde_bandwidth(x: np.ndarray) -> float:
        """
        This fixes a peculiar problem with sns.kdeplot/distplot. Use this to compute the bandwidth and provide it as argument bw. 
        https://stackoverflow.com/questions/61440184/who-is-scott-valueerror-in-seaborn-pairplot-could-not-convert-string-to-floa
        :param x: input (numpy array).
        :return KDE bandwidth computed by Scott's method.
        """
        return statsmodels.nonparametric.bandwidths.bw_scott(x)


class EpsDTransformer:
    """
    Converter between noise magnitude (epsilon) and decay factor (d).
    """
    
    def eps_to_d(self, eps: float) -> float:
        """
        Converts d to epsilon.
        """
        return 1 - 1 / np.sqrt(1 + eps**2)
    
    def d_to_eps(self, d: float) -> float:
        """
        Converts epsilon to d.
        """
        return np.sqrt((1 / (1 - d))**2 - 1)
        

class TauRhoTransformer:
    """
    Converter between perturbation likelihood (tau) and norm bound (rho).
    """
    
    def __init__(self, n_l: int, eps: float):
        self.n_l = n_l
        self.c1 = n_l * np.log(np.sqrt((1 + eps**2) / (2 * np.pi * eps**2)))
        self.c2 = (1 + eps**2) / (2 * eps**2) * np.sqrt(n_l)
        
    def rho_to_logtau(self, rho: float) -> float:
        """
        Converts rho to the logarithm of tau.
        """
        return self.c1 - (rho**2) * self.n_l * self.c2
    
    def logtau_to_rho(self, tau: float) -> float:
        """
        Converts the logarithm of tau to rho.
        """
        return np.sqrt((self.c1 - tau) / self.c2 / self.n_l)

class LogUtil:
    """
    Performs logging of text and images to a new timestamped directory.
    Logs are also printed to stdout. Figures are only shown in notebooks.
    """
    
    _timestamp = lambda: str(int(round(time.time() * 1000)))
    _dirname = "logdir_" + _timestamp()
    _created = False
    
    @staticmethod
    def set_custom_dirname(dirname: str):
        """
        Set the custom name for a logging directory.
        The previous contents of the directory will be deleted!
        """
        LogUtil._dirname = dirname
        LogUtil.ensure_dir_existence()
        for filename in os.listdir(dirname):
            os.remove(os.path.join(dirname, filename))
    
    @staticmethod
    def ensure_dir_existence():
        """
        Creates the directory for logging if it hasn't been created yet.
        """
        if not LogUtil._created:
            LogUtil._created = True
            os.makedirs(LogUtil._dirname, exist_ok=True)
    
    @staticmethod
    def info(msg: str, regular_print: bool = True):
        """
        Log/print a message to <log directory>/log.txt.
        :param msg: message to log and, optionally, print to console.
        :param regular_print: whether to print msg to console.
        """
        LogUtil.ensure_dir_existence()
        if regular_print:
            print(msg)
        with open(os.path.join(LogUtil._dirname, "log.txt"), "a+", encoding="utf-8") as f:
            f.write(f"[time_ms={round(time.time() * 1000)}] {msg}\n")
    
    @staticmethod
    def savefig(prefix: str, pdf: bool = False):
        """
        Saves a figure to the logging directory. Filename is generated based on the current timestamp
        and the provided prefix.
        :param prefix: to be included in the filename.
        :param pdf: if True, save as PDF. If False, save as PNG.
        """
        LogUtil.ensure_dir_existence()
        fname = os.path.join(LogUtil._dirname, "fig_" + prefix + "_" + LogUtil._timestamp() + (".pdf" if pdf else ".png"))
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        LogUtil.info(f"[produced a figure: {fname}]", False)
        
    @staticmethod
    def metacall(fn: Callable, fn_name: str, *args, **kwargs):
        """
        Makes a function call and logs it with concrete argument values.
        :param fn: function to call.
        :param fn_name: the name of the function to be put to the log.
        :param args: *args to be passed to fn.
        :param kwargs: *kwargs to be passed to fn.
        """
        arg_assignments = [repr(arg) for arg in args]
        kwarg_assignments = [f"{kwarg_name}={repr(kwarg_value)}" for kwarg_name, kwarg_value in kwargs.items()]
        LogUtil.info(f"{fn_name}(" + ", ".join(arg_assignments + kwarg_assignments) + ")", False)
        fn(*args, **kwargs)
        
    @staticmethod
    def to_pdf():
        """
        Switches matplotlib backend to 'pdf'.
        """
        matplotlib.use("pdf", warn=False, force=True)
        
