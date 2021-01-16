from typing import *
import copy
from abc import ABC, abstractmethod
import itertools

import numpy as np
import torch.nn
import torch.autograd
from tensorboardX import SummaryWriter

from .ml_util import *
from .datasets import *

class ResBlock(torch.nn.Module):
    def __init__(self, base_map_num: int):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(base_map_num, base_map_num, 3, padding=1),
            torch.nn.BatchNorm2d(base_map_num, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_map_num, base_map_num, 3, padding=1),
            torch.nn.BatchNorm2d(base_map_num, track_running_stats=False),
        )

    def forward(self, x):
        return self.relu(x + self.layer(x))


class Unit(torch.nn.Module):
    """
    Basic convolutional block.
    """
    
    def __init__(self, in_channels: int, out_channels: int, p_dropout: float, unit_type: int = 0):
        """
        Constructs Unit.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param p_dropout: probability of dropout at the end of the layer.
        :param unit_type: 0 (basic choice), 1 (with ResBlocks) or 2 (deeper version of 0).
        """
        super().__init__()
        block = lambda in_ch: [
            torch.nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels, track_running_stats=False),
        ]
        end = lambda: [
            torch.nn.MaxPool2d(kernel_size=2, padding=0),
            torch.nn.Dropout(p_dropout),
        ]
        if unit_type == 0:
            self.model = torch.nn.Sequential(
                *block(in_channels),
                *block(out_channels),
                *end(),
            )
        elif unit_type == 1:
            self.model = torch.nn.Sequential(
                ResBlock(in_channels),
                *block(in_channels),
                *end(),
            )
        elif unit_type == 2:
            self.model = torch.nn.Sequential(
                *block(in_channels),
                *block(out_channels),
                *block(out_channels),
                *end(),
            )
        else:
            raise AssertionError(f"Unexpected unit type {unit_type}")

    def forward(self, x):
        return self.model(x)


class ColoredNet128(torch.nn.Module):
    """
    Simple CNN classifier to process 128x128 images.
    """
    
    def __init__(self, no_classes: int = 2, unit_type: int = 0):
        """
        Constructs ColoredNet128.
        :param no_classes: number of output classes.
        :param unit_type: 0 (basic choice), 1 (with ResBlocks) or 2 (deeper version of 0).
        """
        super().__init__()
        base_map_num = 8
        self.model = torch.nn.Sequential(
            Unit(3,                 1 * base_map_num, 0.2,  unit_type),
            Unit(1 * base_map_num,  2 * base_map_num, 0.25, unit_type),
            Unit(2 * base_map_num,  4 * base_map_num, 0.25, unit_type),
            Unit(4 * base_map_num,  8 * base_map_num, 0.3,  unit_type),
            Unit(8 * base_map_num, 16 * base_map_num, 0.4,  unit_type),
            Flatten(),
            torch.nn.Linear(16 * base_map_num * 4 * 4, no_classes)
        )

    def forward(self, x):
        return self.model(x)


class MNISTNet(torch.nn.Module):
    """
    Simple CNN classifier for 28x28 MNIST images.
    """
    
    def __init__(self, unit_type: int = 0):
        """
        Constructs MNISTNet.
        :param unit_type: 0 (basic choice), 1 (with ResBlocks) or 2 (deeper version of 0).
        """
        super().__init__()
        base_map_num = 8
        self.model = torch.nn.Sequential(
            Lambda(lambda x: torch.nn.functional.pad(input=x, pad=((4,) * 4), mode="constant", value=0)),
            Unit(1,                1 * base_map_num, 0.2, unit_type),
            Unit(1 * base_map_num, 2 * base_map_num, 0.3, unit_type),
            Unit(2 * base_map_num, 4 * base_map_num, 0.4, unit_type),
            Flatten(),
            torch.nn.Linear(4 * base_map_num * 4 * 4, 10)
        )

    def forward(self, x):
        return self.model(x)


class EarlyStoppingMonitor:
    """
    Performs early stopping when the supplied validation metric value starts to decrease.
    """
    
    def __init__(self):
        """
        Constructs EarlyStoppingMonitor.
        """
        self.total_metric_value = -float("inf")
        self.should_stop = False

    def update_metric(self, trainer, new_value: float, epoch: int, disk_backup_filename: str) -> None:
        """
        Sets the flag self.should_stop to True, if the supplied value of the validation metric is not greater
        than the previous one. Otherwise, saves the current trainable parameters of the classifier.
        :param trainer: classifier to work with.
        :param new_value: new validation metric value.
        :param epoch: number of the current epoch.
        :param disk_backup_filename: filename where to save trainable parameters of the classifier if the
            metric still gets better.
        """
        LogUtil.info(f"*** After E{epoch}, validation accuracy = {new_value:.6f} "
                     f"vs. {self.total_metric_value:.6f} on the previous epoch.")
        if new_value > self.total_metric_value:
            LogUtil.info("*** Saving parameters.")
            trainer.save_params()
            trainer.save_params_to_disk(disk_backup_filename)
            self.total_metric_value = new_value
        else:
            LogUtil.info("*** Restoring parameters and halting.")
            trainer.restore_params()
            self.should_stop = True


class Trainer:
    """
    Trainable CNN classifier.
    """
    
    def __init__(self, dataset: str, train_loader_fn: Callable, val_loader_fn: Callable, unit_type: int = 0):
        """
        Constructs Trainer.
        :param dataset: one of 'celeba-128', 'lsun-128', 'mnist', 'none'. This determines network architecture.
        :param train_loader_fn: function that returns a TorchVision loader of (possibly augmented) training data.
        :param val_loader_fn: function that returns a TorchVision loader of validation/test data.
        :param unit_type: unit_type: 0 (basic choice), 1 (with ResBlocks) or 2 (deeper version of 0).
        """
        if dataset in ["celeba-128", "lsun-128"]:
            self.model = ColoredNet128(2, unit_type)
        elif dataset == "mnist":
            self.model = MNISTNet(unit_type)
        elif dataset == "none":
            self.model = None
        else:
            raise RuntimeError("Supported datasets: celeba-128, lsun-128, mnist, none.")
        if self.model is not None:
            self.model = Util.conditional_to_cuda(self.model)
            LogUtil.info(f"{dataset} classifier: {Util.number_of_trainable_parameters(self.model)} trainable parameters")
        self.params = {}
        self.train_loader_fn = train_loader_fn
        self.val_loader_fn = val_loader_fn
        # controls the behavior of predict_with_gradient
        self.misclassification_gradients = True
    
    @torch.no_grad()
    def save_params(self):
        """
        Saves the trainable parameters of the current neural network.
        """
        self.params = copy.deepcopy(self.model.state_dict())

    @torch.no_grad()
    def save_params_to_disk(self, filename: str):
        """
        Saves current trainable parameters to disk.
        :param filename: target filename.
        """
        torch.save(self.model.state_dict(), filename)

    @torch.no_grad()
    def restore_params(self):
        """
        Restores previously saved trainable parameters of the neural network.
        """
        self.model.load_state_dict(self.params)

    @torch.no_grad()
    def restore_params_from_disk(self, filename: str):
        """
        Restores previously saved trainable parameters from disk.
        :param filename: target filename.
        """
        self.params = torch.load(filename, map_location=("cuda:0" if Util.using_cuda else "cpu"))
        self.restore_params()

    def disable_param_gradients(self):
        """
        Disables the gradients of model's parameters. This saves computation time when gradients need to
        be computed w.r.t. the input image only.
        """
        Util.set_param_requires_grad(self.model, False)
            
    @torch.no_grad()
    def random_init(self):
        """
        Performs random initialization.
        """
        def init_weights(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv2d):
                m.weight.normal_(0, 0.1)
                m.bias.normal_(0, 0.1)
            elif isinstance(m, torch.nn.Linear):
                fan_in: int = m.weight.size(0)
                # heuristic
                std = (2 / fan_in) ** 0.5 / 20
                m.weight.uniform_(-std, std)
                if m.bias is not None:
                    m.bias.uniform_(0, 0)
        self.model.apply(init_weights)

    def set_misclassification_gradients(self, value: bool):
        """
        Controls the behavior of predict_with_gradient.
        :param value: if True, predict_with_gradient will return the gradient that can be used to
            perform an untargeted attack. If False, predict_with_gradient returns the gradient of the loss.
        """
        self.misclassification_gradients = value
        
    def predict_with_gradient(self, x: torch.Tensor, true_labels: List[int]) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Predicts, computes loss and gradient.
        If self.misclassification_gradients is True, the gradient points in the direction of a
        possible untargeted attack: ∇_x (max(other_probabilities) - correct_probability)
        If self.misclassification_gradients is False, the gradient is ∇_x (loss).
        :param x: batch of input images. PyTorch gradients of x must be enabled by the caller.
            This method will add the gradient to x.grad.
        :param true_labels: labels that correspond to input images.
        :return: (predictions: LongTensor[batch_dim], losses: FloatTensor[batch_dim]).
        """
        self.model.eval()
        self.disable_param_gradients()
        predictions, losses = [], []
        for i in range(x.shape[0]):
            ps = self.model(x[i].unsqueeze(0))[0].softmax(0)
            if self.misclassification_gradients:
                # loss = undirected misclassification
                mask = torch.ones_like(ps)
                mask[true_labels[i]] = 0
                loss = torch.max(ps * mask) - ps[true_labels[i]]
            else:
                # "loss" = target prediction
                loss = ps[true_labels[i]]
            loss.backward()
            losses += [loss.item()]
            with torch.no_grad():
                predictions += [torch.argmax(ps).item()]
        return (Util.conditional_to_cuda(torch.LongTensor(predictions)),
                Util.conditional_to_cuda(torch.FloatTensor(losses)))
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.LongTensor:
        """
        Predict on a batch of input images.
        :param x: batch of input images.
        :return: LongTensor of predictions.
        """
        self.model.eval()  # set prediction mode
        # predicting separately for each item
        # otherwise, prediction quality will drop due to the behavior of batch normalization
        return Util.conditional_to_cuda(torch.LongTensor([torch.argmax(self.model(x[i].unsqueeze(0)), 1)
                                                          for i in range(x.shape[0])]))

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Returns soft predictions for a batch of input images.
        :param x: batch of input images.
        :return: FloatTensor of soft ("probability") predictions.
        """
        self.model.eval()  # set prediction mode
        # predicting separately for each item
        # otherwise, prediction quality will drop due to the behavior of batch normalization
        return torch.softmax(torch.cat([self.model(x[i].unsqueeze(0)) for i in range(x.shape[0])]), 1)
    
    def fit(self, class_weights: List[float], epochs: int = 2, lr: float = 1e-3, lr_decay: float = 0.1,
            disk_backup_filename: str = "dumped_weights.bin", noise_sigma: float = 0) -> None:
        """
        Trains the classifier using the cross-entropy loss and RMSProp.
        :param class_weights: weights (importance values) of each image class.
        :param epochs: number of epochs. An epoch is a full pass over the train loader.
        :param lr: learning rate.
        :param lr_decay: after each epoch, the learning rate is multiplied by 1 - lr_decay.
        :param disk_backup_filename: filename to dump trainable parameters. Dumping is done once per epoch.
        :param noise_sigma: if positive, Gaussian noise N(0, noise_sigma^2 I) will be added to training images as a form
            of data augmentation. This results in adversarial training according to [Ford, N., Gilmer, J., Carlini, N.,
            & Cubuk, D. (2019). Adversarial examples are a natural consequence of test error in noise.
            arXiv preprint arXiv:1901.10513].
        """
        Util.set_param_requires_grad(self.model, True)
        if not self.params:
            self.random_init()
            self.save_params()
            self.save_params_to_disk(disk_backup_filename)
        else:
            self.restore_params()
        optimizer = torch.optim.RMSprop(self.model.parameters(), weight_decay=1e-6, lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(1 - lr_decay))
        early_stopper = EarlyStoppingMonitor()
        loss_f = torch.nn.CrossEntropyLoss(weight=Util.conditional_to_cuda(torch.FloatTensor(class_weights)))
        writer = SummaryWriter("cnn_training" if noise_sigma == 0 else "cnn_training_noise")

        class EvaluatedPrediction:
            def __init__(self, trainer: Trainer, image: torch.Tensor, label: torch.Tensor):
                if noise_sigma > 0:
                    # augmentation with noise for robust optimization
                    image += Util.conditional_to_cuda(torch.randn(*image.shape)) * noise_sigma
                outputs = trainer.model(image.unsqueeze(0))
                self.loss = loss_f(outputs, label.unsqueeze(0))
                prediction = torch.argmax(outputs, 1)[0]
                self.accuracy = (prediction == label).float().item()

        infinite_val_stream = Util.leakless_cycle(self.val_loader_fn)
        batch_index = 0
        
        for epoch in range(epochs):
            for images, labels in self.train_loader_fn():
                # train
                self.model.train()
                no_train = len(images)
                optimizer.zero_grad()
                loss_sum = 0
                for image, label in zip(images, labels):
                    p = EvaluatedPrediction(self, image, label)
                    (p.loss / no_train).backward()
                    loss_sum += p.loss.item()
                writer.add_scalar("epoch", epoch, batch_index)
                writer.add_scalar("training_loss", loss_sum / no_train, batch_index)
                optimizer.step()

                # validate
                self.model.eval()
                images, labels = next(infinite_val_stream)
                no_val = len(images)
                val_loss_sum = 0
                val_accuracy = 0
                with torch.no_grad():
                    for image, label in zip(images, labels):
                        p = EvaluatedPrediction(self, image, label)
                        val_loss_sum += p.loss.item()
                        val_accuracy += p.accuracy
                writer.add_scalar("validation_loss",     val_loss_sum / no_val, batch_index)
                writer.add_scalar("validation_accuracy", val_accuracy / no_val, batch_index)
                batch_index += 1

            # end of epoch
            scheduler.step()
            self.model.eval()
            with torch.no_grad():
                val_predictions = 0
                items = 0
                for data in self.val_loader_fn():
                    for image, label in zip(*data):
                        val_predictions += EvaluatedPrediction(self, image, label).accuracy
                        items += 1
                early_stopper.update_metric(self, val_predictions / items, epoch, disk_backup_filename)
                if early_stopper.should_stop:
                    break
                    
    def accuracy(self, data_loader_fn: Callable, noise_sigma: float = 0,
                 noise_evaluation_multiplier: int = 1) -> Tuple[float, int]:
        """
        Measures accuracy on the supplied data loader.
        :param data_loader_fn: a function that returns the data loader to be used.
        :param noise_sigma: if positive, Gaussian noise N(0, noise_sigma^2 I) will be added to images
            used in accuracy evaluation.
        :param noise_evaluation_multiplier: evaluation on noise-corrupted images will be done noise_evaluation_multiplier
            times.
        :return: (accuracy, number of images used to calculate it).
        """
        correct = total = 0
        for images, labels in data_loader_fn():
            times = noise_evaluation_multiplier if noise_sigma > 0 else 1
            for i in range(times):
                # measuring corruption robustness with Gaussian noise
                maybe_corrupted_images = images + Util.conditional_to_cuda(torch.randn(*images.shape)) * noise_sigma
                predicted = self.predict(maybe_corrupted_images)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total, total
    
    def measure_robustness(self, perturb_fn: Callable[[torch.FloatTensor, int], torch.FloatTensor],
                           data_loader_fn: Callable, ds: DatasetWrapper, show_images: bool = True) -> Tuple[float, int]:
        """
        Measures robustness on the supplied data loader as accuracy on perturned images.
        :param perturb_fn: function that accepts an image and its true label and perturbs this image.
        :param data_loader_fn: a function that returns the data loader to be used.
        :param ds: DatasetWrapper of the used dataset.
        :param show_images: if True, also shows perturbed images.
        :return: (accuracy, number of images used to calculate it).
        """
        self.model.eval()
        self.disable_param_gradients()
        all_accuracies = []
        for data in data_loader_fn():
            images, labels = data
            perturbed_images, predictions, accuracies = [], [], []
            for image, label in zip(images, labels):
                perturbed_image = perturb_fn(image, label).unsqueeze(0)
                perturbed_images += [perturbed_image]
                outputs = self.model(perturbed_image)
                prediction = torch.argmax(outputs, 1)[0]
                accuracy = (prediction == label).float().item()
                predictions += [prediction]
                accuracies += [accuracy]
            all_accuracies += accuracies
            if show_images:
                Util.imshow_tensors(*perturbed_images, captions=ds.prediction_indices_to_classes(predictions))
        return np.array(all_accuracies).mean(), len(all_accuracies)
    
    def measure_adversarial_severity(self, perturb_fn: Callable[[torch.FloatTensor, int], torch.FloatTensor],
                                     data_loader_fn: Callable, ds: DatasetWrapper,
                                     norm_fn: Callable[[torch.FloatTensor], float],
                                     show_images: bool = True) -> Tuple[float, float, int]:
        """
        Measures adversarial severity (mean norm of minimum adversarial perturbations) on the supplied data loader.
        :param perturb_fn: function that accepts an image and its true label and perturbs this image.
        :param data_loader_fn: a function that returns the data loader to be used.
        :param ds: DatasetWrapper of the used dataset.
        :param norm_fn: function that accepts a PyTorch tensor and returns its norm.
        :param show_images: if True, also shows perturbed images.
        :return: (mean of approximately minimum adversarial perturbations,
            standard deviation of approximately minimum adversarial perturbations,
            number of images used to calculate these values).
        """
        self.model.eval()
        self.disable_param_gradients()
        severities = []
        for data in data_loader_fn():
            images, labels = data
            perturbed_images, predictions = [], []
            for image, label in zip(images, labels):
                perturbed_image = perturb_fn(image, label).unsqueeze(0)
                perturbed_images += [perturbed_image]
                outputs = self.model(perturbed_image)
                prediction = torch.argmax(outputs, 1)[0]
                predictions += [prediction]
                severities += [norm_fn(perturbed_image - image)]
            if show_images:
                Util.imshow_tensors(*perturbed_images, captions=ds.prediction_indices_to_classes(predictions))
        severities = torch.tensor(severities)
        return severities.mean(), severities.std(), len(severities)
        