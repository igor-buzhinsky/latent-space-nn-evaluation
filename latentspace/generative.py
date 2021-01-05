import sys
import os.path
import gc
from typing import *
from abc import ABC, abstractmethod
from importlib import reload
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision

from .ml_util import *
from .datasets import *
from .gan import GAN


class GenerativeModel(ABC):
    """
    Base class for generative models. A generative model can
    (1) generate images from N(0, I)-distributed latent codes;
    (2) encode images into N(0, I)-distributed latent codes.
    """
    
    def __init__(self, resolution: int, latent_dim: int, unique_label: int, ds: DatasetWrapper):
        """
        Constructs GenerativeModel.
        :param resolution: image resolution (resolution == width == height).
        :param latent_dim: number of latent components.
        :param unique_label: class label for which a generative model is created.
        :param ds: DatasetWrapper.
        """
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.unique_label = unique_label
        self.ds = ds
        self.subclass = ds.prediction_indices_to_classes([unique_label])[0]
    
    def destroy(self):
        """
        Free memory.
        """
        pass
    
    def _maybe_detach(self, x: torch.Tensor, detach: bool) -> torch.Tensor:
        """
        Optionally detaches the supplied tensor.
        :x: input tensor.
        :param detach: whether to .detach() the result. 
        """
        return x.detach() if detach else x
    
    @abstractmethod
    def generate(self, no_img: int, detach: bool = False) -> torch.Tensor:
        """
        Generate random images.
        :param no_img: number of images to generate.
        :param detach: whether to .detach() the result.
        :return: batch of generated random images.
        """
        pass
    
    @abstractmethod
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode given images into latent codes.
        :param img: batch of images to encode.
        :return: latent codes for input images.
        """
        pass
    
    @abstractmethod
    def decode(self, latent_code: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """
        Decode given latent codes into images.
        :param latent_code: batch of latent codes to decode.
        :param detach: whether to .detach() the result.
        :return: batch of decoded images.
        """
        pass
    
    def get_sampler(self, batch_size: int = 1):
        """
        Get a data loader for this data class.
        :param batch_size: batch size to be used by the loader.
        :return: data loader (not a function).
        """
        f = lambda: self.ds.get_unaugmented_train_loader(batch_size=batch_size)
        return Util.class_specific_loader(self.unique_label, f)()
    
    @torch.enable_grad()
    def encode_with_gradient_search(self, img: torch.Tensor, no_restarts: int, no_steps: int, lr: float) -> torch.Tensor:
        """
        Implements encoding with a gradient-based approach (4 restarts of Adam).
        """
        all_latent = Util.conditional_to_cuda(torch.empty((img.shape[0], self.latent_dim), dtype=torch.float32))
        loss_f = torch.nn.MSELoss()
        for i in range(img.shape[0]):
            best_loss = np.infty
            for attempt in range(no_restarts):
                latent = Util.optimizable_clone(torch.randn((1, self.latent_dim), dtype=torch.float32))
                optimizer = torch.optim.Adam([latent], weight_decay=0, lr=lr)
                #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
                for j in range(no_steps):
                    optimizer.zero_grad()
                    reconstructed = self.decode(latent, False)
                    loss = loss_f(reconstructed[0], img[i])
                    loss.backward()
                    #print(loss.item())
                    optimizer.step()
                    #scheduler.step()
                #print(loss.item())
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    all_latent[i] = latent.detach()
        #print(f"{best_loss:.3f}")
        return all_latent


class WGAN(GenerativeModel):
    """
    Adapts GAN to GenerativeModel.
    """
    
    def __init__(self, dataset_info: DatasetInfo, unique_label: int, ds: DatasetWrapper):
        """
        Constructs WGAN.
        :param dataset_info: DatasetInfo.
        :param unique_label: label of the data class for which WGAN is created.
        :param ds: DatasetWrapper.
        """
        assert dataset_info == DatasetInfo.MNIST
        self.g = GAN()
        self.g.restore_params_from_disk(f"mnist-gan/dumped_weights_{unique_label}.bin")
        # disable computation of gradients that are not required
        Util.set_param_requires_grad(self.g.generator, False)
        # since discriminator is not used, no need to warm spectral norms
        # self.g.warm_spectral_norms(ds.get_unaugmented_train_loader)
        super().__init__(28, self.g.latent_dim, unique_label, ds)
        
    def generate(self, no_img: int = 1, detach: bool = False) -> torch.Tensor:
        return self._maybe_detach(self.g.generate(no_img), detach)
    
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Implements WGAN.encode with a gradient-based approach (4 restarts of Adam).
        """
        return self.encode_with_gradient_search(img, 4, 40, 0.1)
    
    def decode(self, latent_code: torch.Tensor, detach: bool = True) -> torch.Tensor:
        return self._maybe_detach(self.g.decode(latent_code), detach)
    
    def destroy(self):
        self.g = None
        gc.collect()

        
class BigGAN(GenerativeModel):
    """
    BigGAN wrapper.
    """
    
    def __init__(self, ds: DatasetWrapper, builtin_decay: float = 0.0):
        """
        Constructs BigGAN for several ImageNet classes.
        :param ds: DatasetWrapper.
        :param ds: built-in latent decay (0..1) that makes images more plausible but less diverse.
        """
        super().__init__(128, 120, 0, ds)
        LogUtil.info("*** Loading BigGAN...")
        sys.argv = ("sample.py --dataset I128_hdf5 --shuffle --num_workers 0 --batch_size 1 "
                    "--num_G_accumulations 8 --num_D_accumulations 8 --num_D_steps 1 --G_lr 1e-4 "
                    "--D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 --G_attn 64 --D_attn 64 --G_ch 96 "
                    "--D_ch 96 --G_nl inplace_relu --D_nl inplace_relu --SN_eps 1e-6 --BN_eps 1e-5 "
                    "--adam_eps 1e-6 --G_ortho 0.0 --G_shared --G_init ortho --D_init ortho "
                    "--skip_init --hier --dim_z 120 --shared_dim 128 --ema --ema_start 20000 "
                    "--use_multiepoch_sampler --test_every 2000 --save_every 1000 "
                    "--num_best_copies 5 --num_save_copies 2 --seed 0 --skip_init --G_batch_size 1 "
                    "--use_ema --G_eval_mode --base_root biggan").split(" ")
        from biggan.sample import main, create_sample, decode_sample
        self.create_sample = create_sample
        self.decode_sample = decode_sample
        main("cuda:0" if Util.using_cuda else "cpu")
        self.unique_label = None
        self.builtin_decay = builtin_decay
    
    def configure_label(self, unique_label: int):
        """
        Sets the ImageNet-1k class to generate.
        :param unique_label: ImageNet class label (0..999)
        """
        self.unique_label = unique_label
    
    def generate(self, no_img: int = 1, detach: bool = False) -> torch.Tensor:
        tensors = [self._maybe_detach(self.create_sample([self.unique_label], self.builtin_decay)[0], detach)
                   for _ in range(no_img)]
        return torch.cat(tensors)
    
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Implements BigGAN.encode with a gradient-based approach (4 restarts of Adam).
        """
        return self.encode_with_gradient_search(img, 4, 40, 0.1)
    
    def decode(self, latent_code: torch.Tensor, detach: bool = True) -> torch.Tensor:
        return torch.cat([self._maybe_detach(self.decode_sample(latent_code[i:(i+1)],
                                                                self.unique_label,
                                                                self.builtin_decay), detach)
                          for i in range(latent_code.shape[0])])
    
    def destroy(self):
        gc.collect()
        
    
class PIONEER(GenerativeModel):
    """
    PIONEER [Heljakka, Ari, Arno Solin, and Juho Kannala. "Pioneer networks: Progressively growing generative
    autoencoder." Asian Conference on Computer Vision. Springer, Cham, 2018] model wrapper.
    A copy of PIONEER's sourse code (https://github.com/heljakka/pioneer/) is located in pioneer/src and is
    slightly modified for the purpose of this project.
    """
    
    def __init__(self, dataset_info: DatasetInfo, unique_label: int, ds: DatasetWrapper,
                 spectral_norm_warming_no_images: int = 50):
        """
        Constructs PIONEER. This call is memory-intensive.
        :param dataset_info: DatasetInfo.
        :param unique_label: label of the data class for which WGAN is created.
        :param ds: DatasetWrapper.
        :param spectral_norm_warming_no_images: number of images for spectral norm warming.
            See GAN.warm_spectral_norms.
        """
        
        super().__init__(128, 511, unique_label, ds)
        
        # Model names should end with the number of training iterations. Currently,
        # these numbers are hard-coded and correspond to very concrete models.
        if dataset_info == DatasetInfo.CelebA128Gender:
            pioneer_d = "celeba"
            save_dir = f"pioneer/CelebA-128-{self.subclass}"
            test_path = f"data/celeba/img_align_celeba_{self.subclass}"
            if self.subclass == "male":
                start_iteration = 22452000
            elif self.subclass == "female":
                start_iteration = 20580000
            else:
                raise RuntimeError()
        elif dataset_info == DatasetInfo.LSUN128:
            pioneer_d = "lsun"
            save_dir = f"pioneer/LSUN-128-{self.subclass}"
            test_path = f"data/LSUN/data/LSUN/{self.subclass}_train_lmdb"
            if self.subclass == "bedroom":
                start_iteration = 15268000
            elif self.subclass == "church_outdoor":
                start_iteration = 20212000
            else:
                raise RuntimeError()
        else:
            raise RuntimeError()
        
        model_filename = Path(save_dir) / "checkpoint" / f"{start_iteration}_state"
        if not os.path.isfile(model_filename):
            raise RuntimeError(f"PIONEER model file {model_filename} does not exist!")
        
        max_phase = int(np.round(np.log2(self.resolution))) - 2
        
        # prepare command line arguments
        # in particular, disable random seed resetting with --manual_seed=-1
        LogUtil.info("*** Loading PIONEER...")
        sys.argv = (f"train.py -d {pioneer_d} --start_iteration {start_iteration} --save_dir {save_dir} "
            f"--test_path {test_path} --sample_N=256 --reconstructions_N=0 --interpolate_N=0 "
            f"--max_phase={max_phase} --testonly --no_TB --manual_seed=-1").split(" ")
        LogUtil.info(f"PIONEER's command line arguments: {sys.argv}")
        sys.path.append("./pioneer/src")

        # PIONEER imports
        import config
        self.config = reload(config)
        import evaluate
        self.evaluate = reload(evaluate)
        import model
        self.model = reload(model)
        import data
        self.data = reload(data)
        import utils
        self.utils = reload(utils)
        import train
        self.train = reload(train)
        
        # initialize PIONEER
        train.setup()
        self.session = train.Session()
        self.session.create()
        for model in (self.session.generator, self.session.encoder):
            model.eval()
            # disable computation of gradients that are not required
            Util.set_param_requires_grad(model, False)
        
        # "warm" spectral norms in the encoder
        LogUtil.info(f"*** Spectral norm warming with {spectral_norm_warming_no_images} images...")
        with torch.no_grad():
            dataset = self.get_sampler()
            for i in range(spectral_norm_warming_no_images):
                self.encode_plain(next(dataset)[0])[0]
       
        LogUtil.info("*** Done!")
    
    def destroy(self):
        """
        When PIONEER models are constructed for different classes but only one model is needed at a time,
        this method is useful to free some memory.
        """
        LogUtil.info("*** Cleaning up PIONEER...")
        self.config, self.evaluate, self.model, self.data, self.utils, self.train, self.session = [None] * 7
        gc.collect()
    
    def generate_plain(self, no_img: int = 1, detach: bool = False) -> torch.Tensor:
        """
        Originally, PIONEER generates images from normalized (divided by L2 norm) latent vectors.
        In a multidimensional space, this is approximately the same as taking an N(0, I)-distributed
        vector and dividing it by sqrt(latent_dim). 
        """
        return self.decode_plain(self.utils.normalize(torch.randn(no_img, self.latent_dim)), detach)
    
    def generate(self, no_img: int = 1, detach: bool = False) -> torch.Tensor:
        return self.generate_plain(no_img, detach)
    
    def encode_plain(self, img: torch.Tensor) -> torch.Tensor:
        """
        PIONEER encoder always generates a normalized vector (scaled L2 norm == 1).
        This is not very important in a multidimensional space, since N(0, sigma^2 I)-distributed
        vectors are concentrated around a sphere of radius sigma.
        """
        ex = self.session.encoder(img, self.session.phase, self.session.alpha, self.data.args.use_ALQ).detach()
        return self.utils.split_labels_out_of_latent(ex)[0]
    
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        return self.encode_plain(img) * np.sqrt(self.latent_dim)
    
    def decode_plain(self, latent_code: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """
        Originally, PIONEER decodes images from normalized (divided by L2 norm) latent vectors.
        In a multidimensional space, this is approximately the same as taking an N(0, I)-distributed
        vector and dividing it by sqrt(latent_dim). 
        """
        label = Util.conditional_to_cuda(torch.zeros(latent_code.shape[0], 1, dtype=torch.float32))
        generated = self.session.generator(latent_code, label, self.session.phase, self.session.alpha)
        return self._maybe_detach(generated, detach)
    
    def decode(self, latent_code: torch.Tensor, detach: bool = True) -> torch.Tensor:
        return self.decode_plain(latent_code / np.sqrt(self.latent_dim), detach)
    
