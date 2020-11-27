import numpy as np
import copy
import torch
from typing import *
from tensorboardX import SummaryWriter

from ml_util import *


def init_linear(layer):
    torch.nn.init.xavier_normal_(layer.weight)
    layer.bias.data.zero_()

    
def init_conv(layer):
    torch.nn.init.kaiming_normal_(layer.weight)
    if layer.bias is not None:
        layer.bias.data.zero_()


class SpectralNorm:
    """
    Spectral normalization. Borrowed from https://github.com/heljakka/pioneer/blob/master/src/model.py.
    """
    
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.cuda(async=(args.gpu_count>1))
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)
        return weight_sn, torch.autograd.Variable(u.data)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', torch.nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = torch.autograd.Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + '_u', u)
        setattr(module, name, fn.compute_weight(module)[0])
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)
    return module


class SpectralNormConv2d(torch.nn.Module):
    """
    Convolutional layer with spectral normalization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = torch.nn.Conv2d(*args, **kwargs)
        init_conv(conv)
        self.conv = spectral_norm(conv)

    def forward(self, input):
        return self.conv(input)


class SpectralNormLinear(torch.nn.Module):
    """
    Linear layer with spectral normalization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        linear = torch.nn.Linear(*args, **kwargs)
        init_linear(linear)
        self.linear = spectral_norm(linear)

    def forward(self, input):
        return self.linear(input)
    

class GeneratorUnit(torch.nn.Module):
    """
    Basic convolutional block of a generator.
    """
    
    def __init__(self, in_channels: int, out_channels: int, nop_end: bool = False):
        super().__init__()
        self.model = torch.nn.Sequential(
            Upsample(),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            NopLayer() if nop_end else torch.nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)
        

class MNISTGenerator(torch.nn.Module):
    """
    MNIST generator.
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        base_map_num = 8
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 4 * base_map_num * 4 * 4),
            Lambda(lambda x: x.view(x.shape[0], 4 * base_map_num, 4, 4)),
            GeneratorUnit(4 * base_map_num, 2 * base_map_num),
            GeneratorUnit(2 * base_map_num, 1 * base_map_num),
            GeneratorUnit(1 * base_map_num, 1, nop_end=True),
            # crop:
            Lambda(lambda x: x[:, :, 2:-2, 2:-2]),
            torch.nn.Sigmoid(),
            Lambda(lambda x: (x - 0.5) * 2),
        )

    def forward(self, x):
        return self.model(x)
    

class DiscriminatorUnit(torch.nn.Module):
    """
    Basic convolutional block of a discriminator.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            SpectralNormConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            SpectralNormConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, padding=0)
        )

    def forward(self, x):
        return self.model(x)
    

class MNISTDiscriminator(torch.nn.Module):
    """
    MNIST discriminator.
    """
    
    def __init__(self):
        super().__init__()
        base_map_num = 12
        self.model = torch.nn.Sequential(
            Lambda(lambda x: torch.nn.functional.pad(x, pad=((2,) * 4), mode="constant", value=-1)),
            DiscriminatorUnit(1,                1 * base_map_num),
            DiscriminatorUnit(1 * base_map_num, 2 * base_map_num),
            DiscriminatorUnit(2 * base_map_num, 4 * base_map_num),
            Flatten(),
            SpectralNormLinear(4 * base_map_num * 4 * 4, 1)
        )

    def forward(self, x):
        return self.model(x)
    

class GAN:
    """
    Wasserstein GAN (WGAN). Only MNIST is supported. The discriminator is made Lipschitz-continuous
    with spectral normalization.
    """
    
    def __init__(self):
        """
        Constructs GAN.
        """
        self.latent_dim = 64
        self.generator = Util.conditional_to_cuda(MNISTGenerator(self.latent_dim))
        self.discriminator = Util.conditional_to_cuda(MNISTDiscriminator())
        self.params = None
        LogUtil.info(f"generator: {Util.number_of_trainable_parameters(self.generator)} trainable parameters")
        LogUtil.info(f"discriminator: {Util.number_of_trainable_parameters(self.discriminator)} trainable parameters")
        
    @torch.no_grad()
    def save_params(self):
        """
        Saves the trainable parameters of the current WGAN.
        """
        self.params = (copy.deepcopy(self.generator.state_dict()),
                       copy.deepcopy(self.discriminator.state_dict()))

    @torch.no_grad()
    def save_params_to_disk(self, filename: str):
        """
        Saves current trainable parameters to disk. This is done with pickle.
        :param filename: target filename.
        """
        Util.dump_model(filename, (copy.deepcopy(self.generator.state_dict()),
                                   copy.deepcopy(self.discriminator.state_dict())))

    @torch.no_grad()
    def restore_params(self):
        """
        Restores previously saved trainable parameters of the WGAN.
        """
        self.generator.load_state_dict(self.params[0])
        self.discriminator.load_state_dict(self.params[1])

    @torch.no_grad()
    def restore_params_from_disk(self, filename: str):
        """
        Restores previously saved trainable parameters from disk. This is done with pickle.
        :param filename: target filename.
        """
        self.params = Util.read_model(filename)
        self.restore_params()

    @torch.no_grad()
    def random_init(self):
        """
        Performs random initialization.
        """
        self.generator.apply(Util.init_weights)
        self.discriminator.apply(Util.init_weights)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent vectors into images.
        :param x: batch of latent vectors.
        :return: batch of generated images.
        """
        return self.generator(x)
    
    @torch.no_grad()
    def generate(self, no_samples: int) -> torch.Tensor:
        """
        Generated new images.
        :param no_samples: nimber of images to generate.
        :return: batch of generated images.
        """
        return self.generator(Util.conditional_to_cuda(torch.randn(no_samples, self.latent_dim)))
    
    @torch.no_grad()
    def warm_spectral_norms(self, train_loader_fn: Callable):
        """
        The currently used implementation of spectral norms has some parameters that are not saved to disk.
        As a result, the discriminator is in an invalid state after it is loaded from disk.
        Running spectral normalization ("warming" spectral norms) on a number of images returns it to shape.
        :param train_loader_fn: loader-returning function to be used for warming.
        """
        data_sampler = iter(train_loader_fn())
        while True:
            try:
                self.discriminator(next(data_sampler)[0])
            except StopIteration:
                break
        
    def fit(self, train_loader_fn: Callable, epochs: int = 2, lr: float = 1e-3, n_critic: int = 5,
            disk_backup_filename: str = "dumped_weights.bin"):
        """
        Trains this WGAN.
        :param train_loader_fn: loader-returning function to generate training data.
        :param epochs: number of epochs. An epoch is a full pass over the train loader.
        :param lr: learning rate.
        :param n_critic: number of steps to train the discriminator (a.k.a. critic) per each training
            step of the generator. In WGANs, it is OK to make it large.
        :param disk_backup_filename: filename to dump trainable parameters. Dumping is done once per epoch.
        """
        Util.set_param_requires_grad(self.generator, True)
        Util.set_param_requires_grad(self.discriminator, True)
        if not self.params:
            self.random_init()
            self.save_params()
            self.save_params_to_disk(disk_backup_filename)
        else:
            self.restore_params()
        
        g_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        d_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)
        self.generator.train()
        self.discriminator.train()
        writer = SummaryWriter("gan_training")
        batch_index = 0
        
        for epoch in range(epochs):
            data_sampler = iter(train_loader_fn())
            while True:
                # preload real batches
                real_batches = []
                for i in range(n_critic):
                    try:
                        real_data, _ = next(data_sampler)
                        real_batches.append(real_data)
                    except StopIteration:
                        # for simplicity, omitting the last incomplete sequence of batches
                        break
                if len(real_batches) != n_critic:
                    # next epoch
                    break
            
                batch_size = real_batches[0].shape[0]
            
                # train
                d_optimizer.zero_grad()
                for i in range(n_critic):
                    real_data = real_batches[i]
                    fake_data = self.generator(Util.conditional_to_cuda(torch.randn(batch_size, self.latent_dim)))
                    loss1 = self.discriminator(real_data).mean()
                    loss2 = self.discriminator(fake_data).mean()
                    discriminator_loss = -(loss1 - loss2)
                    discriminator_loss.backward()
                    d_optimizer.step()
                
                g_optimizer.zero_grad()
                fake_data = self.generator(Util.conditional_to_cuda(torch.randn(batch_size, self.latent_dim)))
                generator_loss = -self.discriminator(fake_data).mean()
                #generator_loss = (fake_data - real_data).abs().mean()
                generator_loss.backward()
                g_optimizer.step()
                
                # eval
                with torch.no_grad():
                    writer.add_scalar("discriminator_loss", discriminator_loss.detach(), batch_index)
                    writer.add_scalar("generator_loss", generator_loss.detach(), batch_index)
                    writer.add_scalar("epoch", epoch, batch_index)
                    if batch_index % 20 == 0:
                        writer.add_images("generated_batch", fake_data.detach().clamp(-1, 1), batch_index)
                        writer.add_images("real_batch", real_data.detach(), batch_index)
                batch_index += 1
        
