import torch
import torchvision
from typing import *
import seaborn as sns

from .ml_util import *
from .generative import GenerativeModel
from .cnn import Trainer


class RandomPerturbationStatistician:
    """
    Measures classification performance on images with added latent noise (local latent noise accurcy - LLNA).
    Also shows these noisy images.
    """
    
    def __init__(self, gm: GenerativeModel, classifiers: List[Trainer], no_images: int,
                 perturbations_per_image: int, visualize: bool, epsilons: List[float]):
        """
        Constructs RandomPerturbationStatistician.
        :param gm: generative model to use.
        :param classifiers: list of clasifiers for which to perform evaluation.
        :param no_images: number of images for which to perform local evaluation.
        :param perturbations_per_image: number of perturbation to perform for each image and epsilon.
        :param visualize: if True, produce images of original and perturbed images.
        :param epsilons: list of noise magnitudes (non-negative numbers). Evaluation will be performed for each of them.
        """
        self.gm = gm
        self.classifiers = classifiers
        self.no_images = no_images
        self.perturbations_per_image = perturbations_per_image
        self.epsilons = epsilons
        self.visualize = visualize
        self.stat_len = 2 + len(epsilons)
    
    @torch.no_grad()
    def process(self):
        """
        Does actual processing.
        """
        dataset = self.gm.get_sampler()
        for i_img in range(self.no_images):
            orig_img, labels = next(dataset)
            latent = self.gm.encode(orig_img)
            reconstr_img = self.gm.decode(latent)
            correct = torch.zeros(len(self.classifiers), self.stat_len)
            for i_pert in range(self.perturbations_per_image):
                all_img = [orig_img, reconstr_img]
                for eps in self.epsilons:
                    perturbation = Util.conditional_to_cuda(torch.randn(1, self.gm.latent_dim))
                    perturbed = (latent + perturbation * eps) / np.sqrt(1 + eps**2)
                    reconstr_pert_img = self.gm.decode(perturbed)
                    all_img += [reconstr_pert_img]
                # predict
                predictions = [c.predict(torch.cat(all_img)) for c in self.classifiers]
                if self.visualize:
                    classes = [self.gm.ds.prediction_indices_to_printed_classes(p) for p in predictions]
                    joined = list("\n".join(t) for t in zip(*classes))
                    Util.imshow_tensors(*all_img, captions=joined)
                # count statistics
                for j in range(self.stat_len):
                    for k in range(len(self.classifiers)):
                        correct[k, j] += (predictions[k][j] == labels[0]).item()
            LogUtil.info(f"Results for all {len(self.classifiers)} classifiers one by one:")
            for j, s in enumerate(["initial", "reconstructed"] + [f"perturbed({eps:.2f})" for eps in self.epsilons]):
                s = f"{s:15s} |"
                for i, c in enumerate(self.classifiers):
                    rate = f"{correct[i, j] / self.perturbations_per_image * 100:.2f}"
                    s += f"   c{i}: {rate:>6}%"
                LogUtil.info(s)


def get_get_gradient(classifier: Trainer, label: int, vector_transform_1: Callable, vector_transform_2: Callable,
                     grad_transform: Callable):
    """
    Produces custom "get_gradient" functions that are accepted by Adversaries.
    (This is not an example of good object-oriented design.)
    :param classifier: classifier for which to get gradient function.
    :param label: true label of the current image (int).
    :param vector_transform_1: vector transformation used to parameterize get_gradient (best to explain with the code below).
    :param vector_transform_2: vector transformation used to parameterize get_gradient (best to explain with code below).
    :param grad_transform: vector transformation used to parameterize get_gradient (best to explain with code below).
    """
    def get_gradient(vector: torch.Tensor) -> Tuple[torch.Tensor, float]:
        vector_g = Util.optimizable_clone(vector_transform_1(vector))
        image = vector_transform_2(vector_g)
        _, loss = classifier.predict_with_gradient(image, [label])
        return grad_transform(vector_g.grad), loss.item()
    return get_gradient


def get_conventional_perturb(classifier: Trainer, adversary: Adversary) -> Callable:
    """
    Produces "perturb" functions that are accepted by Trainers (classifiers).
    Perturbations are performed in the original image space.
    :param classifier: classifier that will accept the resulting function.
    :param adversary: Adversary that will be used by the resulting function.
    :return: a "perturb" function as described above.
    """
    def perturb(image: torch.Tensor, true_label: int) -> torch.Tensor:
        get_gradient = get_get_gradient(classifier, true_label, lambda x: x.view(1, *image.shape),
                                        lambda x: x, lambda x: x.view(1, -1))
        return adversary.perturb(image.view(1, -1), get_gradient).view(*image.shape)
    return perturb

    
class AdversarialGenerator:
    """
    Generates adversarial images with the provided adversary, shows them and computes peturbation statistics.
    """
    
    def __init__(self, gm: GenerativeModel, classifiers: List[Trainer], use_generated_images: bool,
                 decay_factor: float, label_printer: Optional[Callable] = None):
        """
        Constructs AdversarialGenerator.
        :param gm: generative model to use.
        :param classifiers: list of clasifiers for which to perform evaluation.
        :param use_generated_images: when True, calculate generation-based metrics.
            When False, calculate reconstruction-based methods.
        :param decay_factor: decay factor: Float 0..1. The latent code of the image is multiplied by 1 - decay_factor
            prior to adversarial search.
        """
        self.gm = gm
        self.classifiers = classifiers
        for c in classifiers:
            c.disable_param_gradients()
        self.use_generated_images = use_generated_images
        self.decay_factor = decay_factor
        self._clear_stat()
        self.label_printer = label_printer
        
    def set_generative_model(self, gm: GenerativeModel):
        """
        Sets new generative model to be used. This is useful when models are memory-intensive (cannot be loaded at once),
        but statistics needs to be calculated across all models (image classes).
        :param gm: generative model to use.
        """
        self.gm = gm
    
    def _clear_stat(self):
        """
        Initializes (when called first time) or clears accumulated statistics.
        """
        create = lambda: [[] for c in self.classifiers]
        self.recorded_original_l2_norms = create()
        self.recorded_original_l1_norms = create()
        self.recorded_latent_norms = create()
        self.recorded_latent_norm_diffs = create()
        self.recorded_clean_successes = create()
        self.recorded_reconstructed_successes = create()
        self.recorded_decayed_successes = create()
        self.recorded_modified_successes = create()
    
    def _join_predictions(self, img: torch.Tensor, classifiers: List[Trainer]):
        """
        Convenience function.
        """
        predictions = [c.predict(img) for c in classifiers]
        if self.label_printer is None:
            label_printer = lambda x: self.gm.ds.prediction_indices_to_printed_classes(x)[0]
        else:
            label_printer = self.label_printer
        str_classes = [label_printer(p) for p in predictions]
        return "\n".join(str_classes), [x.item() for x in predictions]
    
    def _l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scaled L2 norm of x.
        """
        return x.norm() / np.sqrt(x.numel())
    
    def _l1_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scaled L1 norm of x.
        """
        return x.norm(1) / x.numel()
    
    def print_stats(self, plot: bool = False, print_norm_statistics: bool = True):
        """
        Prints/logs the accumulated statistics. Optionally, plots histograms of metrics computed for adversarial perturbations.
        These are the following metrics:
        - Accuracies on original/generated/reconstructed/perturbed images.
        - ║Δl║ (scaled L2): scaled L2 norms of latent perturbations.
        - ║l + Δl║ - ║l║ (scaled L2): increase of scaled L2 norm after the initial (decayed) images is replaced with an
            adversarial image.
        - ║Δx║ (scaled L2): scaled L2 norm of found adversarial perturbations in the original space.
        - ║Δx║ (scaled L1): scaled L1 norm of found adversarial perturbations in the original space.
        :param plot: whether to plot histograms for each of the aforementioned metrics.
        :param print_norm_statistics: whether to print norm statistics.
        """
        # classification accuracy
        pairs = []
        if self.use_generated_images:
            pairs += [(self.recorded_reconstructed_successes, "    generated")]
        else:
            pairs += [(self.recorded_clean_successes,         "     original"),
                      (self.recorded_reconstructed_successes, "reconstructed")]
        pairs     += [(self.recorded_decayed_successes,       "      decayed")]
        pairs     += [(self.recorded_modified_successes,      "    perturbed")]
        for stat_list, msg in pairs:
            for j in range(len(self.classifiers)):
                accuracy = np.array(stat_list[j]).mean()
                LogUtil.info(f"Accuracy of classifier {j} on {len(stat_list[j])} {msg} images: {accuracy * 100:.2f}%")
                
        # perturbation norms
        iter_list = [(self.recorded_latent_norms,      "            ||dl|| (scaled L2)", r'$||\Delta l||_2^s$'),
                     (self.recorded_latent_norm_diffs, "||l + dl|| - ||l|| (scaled L2)", r'$||l + \Delta l||_2^s - ||l||_2^s$'),
                     (self.recorded_original_l2_norms, "            ||dx|| (scaled L2)", r'$||\Delta x||_2^s$'),
                     (self.recorded_original_l1_norms, "            ||dx|| (scaled L1)", r'$||\Delta x||_1~/~n_I$')]
        if print_norm_statistics:
            for i, (norm_list, console_str, _) in enumerate(iter_list):
                for j in range(len(self.classifiers)):
                    norms = torch.tensor(norm_list[j])
                    qs = np.quantile(norms, np.linspace(0, 1, 5))
                    LogUtil.info(f"For classifier {j} and {len(norms)} images, {console_str:8s}: "
                                 f"mean={norms.mean():.5f}, std={norms.std():.5f}, "
                                 f"Q0={qs[0]:.5f}, Q1={qs[1]:.5f}, Q2={qs[2]:.5f}, Q3={qs[3]:.5f}, Q4={qs[4]:.5f}")
        if plot:
            fig, axarr = plt.subplots(1, 4, figsize=(15, 2.0))
            for i, (norm_list, _, plot_str) in enumerate(iter_list):
                LogUtil.metacall(axarr[i].title.set_text, f"axarr[{i}].title.set_text", plot_str)
                for j in range(len(self.classifiers)):
                    norms = torch.tensor(norm_list[j])
                    #axarr[i].title.set_text(plot_str)
                    #sns.distplot(norms, hist=True, kde=True, ax=axarr[i])
                    LogUtil.metacall(sns.distplot, "sns.distplot", norms, hist=False, kde=True, ax=axarr[i],
                                     kde_kws={"bw": Util.get_kde_bandwidth(norms.numpy())})
            LogUtil.savefig("advgen_stats", True)
            plt.show()
            plt.close()
    
    def generate(self, adversary: Adversary, no_images: int, show_perturbations: bool = False, clear_stat: bool = True,
                 produce_images: bool = True):
        """
        Does actual processing.
        :param adversary: Adversary to be used (e.g., PGDAdversary).
        :param no_images: the number of images for which to perform adversarial search.
        :param show_perturbations: if True, shows not only adversarial examples, but the corresponding perturbations in
            the original image space.
        :param clear_stat: whether to clear statistics prior to adversarial search.
        :param produce_images: whether to actually produce all images.
        """
        if clear_stat:
            self._clear_stat()
        result = ImageSet(1)
        nrow = 3 + len(self.classifiers)
        if show_perturbations:
            nrow += 1 + len(self.classifiers)
        if self.use_generated_images:
            nrow -= 1
        else:
            dataset = self.gm.get_sampler(batch_size=1)

        for _ in range(no_images):
            label = self.gm.unique_label
            with torch.no_grad():
                if self.use_generated_images:
                    latent_code = Util.conditional_to_cuda(torch.randn(1, self.gm.latent_dim))
                    reconstr_img = self.gm.decode(latent_code)
                    # dummy copy
                    orig_img = reconstr_img
                else:
                    orig_img, label_ = next(dataset)
                    assert label_[0] == label
                    latent_code = self.gm.encode(orig_img)
                    reconstr_img = self.gm.decode(latent_code)
                # decay
                latent_code *= 1 - self.decay_factor
                reconstr_img_decayed = self.gm.decode(latent_code)
            
            images = [orig_img, reconstr_img, reconstr_img_decayed]
            list_of_tuples = [self._join_predictions(img, self.classifiers) for img in images]
            str_predictions = [x[0] for x in list_of_tuples]
            predictions = [x[1] for x in list_of_tuples]
            
            if show_perturbations:
                images += [reconstr_img_decayed - reconstr_img]
                str_predictions += [""]
            
            for i, c in enumerate(self.classifiers):
                self.recorded_clean_successes[i]         += [predictions[0][i] == label]
                self.recorded_reconstructed_successes[i] += [predictions[1][i] == label]
                self.recorded_decayed_successes[i]       += [predictions[2][i] == label]
                get_gradient = get_get_gradient(c, label, lambda x: x,
                                                lambda x: self.gm.decode(x, False), lambda x: x)
                perturbed_latent_code = adversary.perturb(latent_code, get_gradient)
                
                # record statistics
                with torch.no_grad():
                    reconstr_img_perturbed = self.gm.decode(perturbed_latent_code)
                    images += [reconstr_img_perturbed]
                    new_str_predictions, new_predictions = self._join_predictions(reconstr_img_perturbed, [c])
                    self.recorded_modified_successes[i] += [new_predictions[0] == label]
                    str_predictions += [new_str_predictions]
                    diff = reconstr_img_perturbed - reconstr_img_decayed
                    self.recorded_original_l2_norms[i] += [self._l2_norm(diff)]
                    self.recorded_original_l1_norms[i] += [self._l1_norm(diff)]
                    self.recorded_latent_norms[i]      += [self._l2_norm(perturbed_latent_code - latent_code)]
                    self.recorded_latent_norm_diffs[i] += [self._l2_norm(perturbed_latent_code) - self._l2_norm(latent_code)]
                    if show_perturbations:
                        images += [diff]
                        str_predictions += [""]

            if self.use_generated_images:
                # remove dummy items
                images = images[1:]
                str_predictions = str_predictions[1:]
            if produce_images:
                result.append(images, str_predictions)
                result.maybe_show(nrow=nrow)

        if produce_images:
            result.maybe_show(True, nrow=nrow)
