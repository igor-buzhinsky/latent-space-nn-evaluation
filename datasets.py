import torch
import torchvision
import torchvision.transforms as transforms
import resource
from abc import ABC, abstractmethod
from typing import *

from ml_util import *


# this may prevent an error in DataLoader with num_workers > 0
resource.setrlimit(resource.RLIMIT_NOFILE, (2048 * 2, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))

# norm bounds for conventional robustness evaluation
MNIST_L2_UPPER_BOUND = 0.6
OTHER_L2_UPPER_BOUND = 0.05
MNIST_LINF_UPPER_BOUND = 1.2
OTHER_LINF_UPPER_BOUND = 0.1


class DatasetWrapper(ABC):
    """
    Base class for dataset wrappers. These classes specify several loading procedures with customizable
    batch sizes, randomization, transformations (including data augmentation).
    Class labels are given as integers (starting from 0) and string captions.
    """
    
    train_batch_size = 32
    test_batch_size = 4
    
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(Util.conditional_to_cuda),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    
    def __init__(self, img_size: int, classes: List[str], printed_classes: List[str], test_transform, train_transform):
        """
        Constructs a DatasetWrapper.
        :param img_size: image resolution (img_size == width == height).
        :param classes: list of full class names.
        :param printed_classes: list of possibly shortended class names (to be printed on top of images).
        :param test_transform: torchvision.transforms transformation for test data.
        :param train_transform: torchvision.transforms transformation for train data.
        """
        self.img_size = img_size
        self.classes = classes
        self.printed_classes = printed_classes
        self.test_transform = test_transform
        self.train_transform = train_transform
        # for datasets producing elements of a single label
        self.unique_label = None
        
    def preprocess_batch(self, xs):
        """
        Ensures that each image in the batch is paired with a single integer label.
        :param xs: batch.
        :return: transformed xs.
        """
        if self.unique_label is None:
            labels = [self.get_label(x[1]) for x in xs]
        else:
            labels = [self.unique_label] * len(xs)
        return torch.stack([x[0] for x in xs]), Util.conditional_to_cuda(torch.tensor(labels))
    
    def get_label(self, x):
        """
        Get the actual (possibly projected) label from the one given by the dataset.
        :param x: original label.
        :return: possibly transformed label. 
        """
        return x
    
    def prediction_indices_to_classes(self, indices: Iterable):
        """
        :param indices: input class indices.
        :return: full class names, to be used in model filenames.
        """
        return [self.classes[int(i)] for i in indices]
    
    def prediction_indices_to_printed_classes(self, indices: Iterable):
        """
        :param indices: input class indices.
        :return: short class names, to be printed on top of images.
        """
        return [self.printed_classes[int(i)] for i in indices]
    
    def get_loader_(self, torchvision_dataset, batch_size: int, shuffle: bool):
        """
        Get a torch.utils.data.DataLoader for this dataset.
        :param torchvision_dataset: TorchVision dataset.
        :param batch_size: batch size.
        :param shuffle: whether to shuffle.
        :return: torch.utils.data.DataLoader.
        """
        return torch.utils.data.DataLoader(torchvision_dataset, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=0, drop_last=True,
                                           collate_fn=self.preprocess_batch)
    
    def get_unaugmented_train_loader(self, batch_size: int = None, shuffle: bool = True):
        """
        Get a train data loader without data augmentation.
        :param batch_size: batch size.
        :param shuffle: whether to shuffle.
        :return: torch.utils.data.DataLoader.
        """
        if batch_size is None:
            batch_size = self.train_batch_size
        return self.get_loader_(self.unaugmented_trainset, batch_size, shuffle)
    
    def get_train_loader(self, batch_size: int = None, shuffle: bool = True):
        """
        Get a train data loader with data augmentation.
        :param batch_size: batch size.
        :param shuffle: whether to shuffle.
        :return: torch.utils.data.DataLoader.
        """
        if batch_size is None:
            batch_size = self.train_batch_size
        return self.get_loader_(self.trainset, batch_size, shuffle)
    
    def get_test_loader(self, batch_size: int = None, shuffle: bool = True):
        """
        Get a test data loader without data augmentation.
        :param batch_size: batch size.
        :param shuffle: whether to shuffle.
        :return: torch.utils.data.DataLoader.
        """
        if batch_size is None:
            batch_size = self.test_batch_size
        return self.get_loader_(self.testset, batch_size, shuffle)
    

class MNISTData(DatasetWrapper):
    """
    MNIST dataset wrapper. If the dataset is missing, it will be downloaded automatically.
    Output images are 3-dimensional but essentially grayscale.
    """
    
    def __init__(self):
        """
        Constructs MNISTData.
        """
        size = 28
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(Util.conditional_to_cuda),
            transforms.Normalize((0.5,), (0.5,))
        ])
        labels = tuple([str(i) for i in range(10)])
        super().__init__(size, labels, labels, base_transform,
            transforms.Compose([
                transforms.RandomAffine(20),
                base_transform,
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            ]),
        )
        self.trainset = torchvision.datasets.MNIST("mnist", train=True, download=True,
                                                   transform=self.train_transform)
        self.unaugmented_trainset = torchvision.datasets.MNIST("mnist", train=True, download=True,
                                                               transform=self.test_transform)
        self.testset = torchvision.datasets.MNIST("mnist", train=False, download=True,
                                                  transform=self.test_transform)
    

class CelebAData(DatasetWrapper):
    """
    CelebA dataset wrapper. If the dataset is missing, it will be downloaded automatically.
    Images are center-cropped as resized to 128x128.
    """
    
    def __init__(self, attribute_index: int):
        """
        Constructs CelebAData.
        :param attribute_index: CelebA attribute index (non-negative integer), e.g., 20 = 'Male'.
        """
        size = 128
        attributes = {}
        # CelebA attributes: index | name | mean[-1..1]
        attributes[0]  = ("5_o_Clock_Shadow", -0.778)
        attributes[1]  = ("Arched_Eyebrows", -0.466)
        attributes[2]  = ("Attractive", 0.025)
        attributes[3]  = ("Bags_Under_Eyes", -0.591)
        attributes[4]  = ("Bald", -0.955)
        attributes[5]  = ("Bangs", -0.697)
        attributes[6]  = ("Big_Lips", -0.518)
        attributes[7]  = ("Big_Nose", -0.531)
        attributes[8]  = ("Black_Hair", -0.521)
        attributes[9]  = ("Blond_Hair", -0.704)
        attributes[10] = ("Blurry", -0.898)
        attributes[11] = ("Brown_Hair", -0.59)
        attributes[12] = ("Bushy_Eyebrows", -0.716)
        attributes[13] = ("Chubby", -0.885)
        attributes[14] = ("Double_Chin", -0.907)
        attributes[15] = ("Eyeglasses", -0.87)
        attributes[16] = ("Goatee", -0.874)
        attributes[17] = ("Gray_Hair", -0.916)
        attributes[18] = ("Heavy_Makeup", -0.226)
        attributes[19] = ("High_Cheekbones", -0.09)
        attributes[20] = ("Male", -0.166)
        attributes[21] = ("Mouth_Slightly_Open", -0.033)
        attributes[22] = ("Mustache", -0.917)
        attributes[23] = ("Narrow_Eyes", -0.77)
        attributes[24] = ("No_Beard", 0.67)
        attributes[25] = ("Oval_Face", -0.432)
        attributes[26] = ("Pale_Skin", -0.914)
        attributes[27] = ("Pointy_Nose", -0.445)
        attributes[28] = ("Receding_Hairline", -0.84)
        attributes[29] = ("Rosy_Cheeks", -0.869)
        attributes[30] = ("Sideburns", -0.887)
        attributes[31] = ("Smiling", -0.036)
        attributes[32] = ("Straight_Hair", -0.583)
        attributes[33] = ("Wavy_Hair", -0.361)
        attributes[34] = ("Wearing_Earrings", -0.622)
        attributes[35] = ("Wearing_Hat", -0.903)
        attributes[36] = ("Wearing_Lipstick", -0.055)
        attributes[37] = ("Wearing_Necklace", -0.754)
        attributes[38] = ("Wearing_Necktie", -0.855)
        attributes[39] = ("Young", 0.547)
        if attribute_index == 20:
            a_no, a_yes = "female", "male"
        elif attribute_index == 39:
            a_no, a_yes = "old", "young"
        else:
            a_no, a_yes = "+" + attributes[i][0], "-" + attributes[i][0]
        self.attribute_index = attribute_index
        
        labels = (a_no, a_yes)
        super().__init__(size, labels, labels,
            transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                DatasetWrapper.base_transform
            ]),
            transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.RandomOrder([
                    transforms.RandomHorizontalFlip(),
                    #transforms.RandomGrayscale(p=0.1),
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2)], 0.7),
                    transforms.RandomApply([transforms.RandomAffine(25, translate=(0.1, 0.1), scale=(0.9, 1.1))], 0.7),
                ]),
                DatasetWrapper.base_transform,
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            ])
        )
        self.trainset = torchvision.datasets.CelebA(root='./data', split="train", download=True,
                                                    transform=self.train_transform)
        self.unaugmented_trainset = torchvision.datasets.CelebA(root='./data', split="train", download=True,
                                                                transform=self.test_transform)
        self.testset = torchvision.datasets.CelebA(root='./data', split="valid", download=True,
                                                   transform=self.test_transform)

    def get_label(self, x):
        # project labels to the specified attribute
        return x[self.attribute_index]
    

class LSUNData(DatasetWrapper):
    """
    LSUN dataset wrapper. If the dataset is missing, it will be downloaded automatically (???).
    Images are center-cropped as resized to 128x128.
    """
    
    def __init__(self, unique_label: int = None):
        """
        Constructs LSUNData.
        :param unique_label: 0 (bedroom), 1 (church outdoor) or None. If None, then items of both labels will be produced.
        """
        size = 128
        labels =         ("bedroom", "church_outdoor")
        printed_labels = ("bedroom", "outdoor")
        #("bedroom", "bridge", "church_outdoor", "classroom", "conference_room", "dining_room",
        # "kitchen", "living_room", "restaurant", "test", "tower")
        super().__init__(size, labels, printed_labels,
            transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                DatasetWrapper.base_transform
            ]),
            transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.RandomOrder([
                    transforms.RandomHorizontalFlip(),
                    #transforms.RandomGrayscale(p=0.1),
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2)], 0.7),
                    transforms.RandomApply([transforms.RandomAffine(25, translate=(0.1, 0.1), scale=(0.9, 1.1))], 0.7),
                ]),
                DatasetWrapper.base_transform,
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            ])
        )
        
        def mnist_class_dataset(train: bool, transform):
            dirname = f"./data/LSUN/{labels[unique_label]}_{'train' if train else 'val'}_lmdb/"
            return torchvision.datasets.LSUNClass(root=dirname, transform=transform)
        
        if unique_label is None:
            self.nested_datasets = [LSUNData(i) for i in range(len(labels))]
            self.trainset = None
            self.unaugmented_trainset = None
            self.testset = None
        else:
            self.trainset =             mnist_class_dataset(True, self.train_transform)
            self.unaugmented_trainset = mnist_class_dataset(True, self.test_transform)
            self.testset =              mnist_class_dataset(False, self.test_transform)
        
        self.unique_label = unique_label
        
    @staticmethod
    def merge_loaders_(parent_loader_1_fn: Callable, parent_loader_2_fn: Callable, batch_size: int, shuffle: bool) -> Callable:
        """
        Merges two dataset loaders. Batch size is preserved.
        Labels are replaced with 0 (first loader) and 1 (second loader).
        Items from both are cycled until the end of any of them is reached.
        """
        data_generator1 = iter(parent_loader_1_fn(batch_size, shuffle))
        data_generator2 = iter(parent_loader_2_fn(batch_size, shuffle))
        while True:
            # load
            try:
                items1, labels1 = next(data_generator1)
                items2, labels2 = next(data_generator2)
            except StopIteration:
                return
            all_items = torch.cat([items1, items2])
            all_labels = torch.cat([labels1, labels2])
            # shuffle
            if shuffle:
                perm = torch.randperm(batch_size * 2)
                all_items = all_items[perm]
                all_labels = all_labels[perm]
            # spit out
            yield all_items[:batch_size], all_labels[:batch_size]
            yield all_items[batch_size:], all_labels[batch_size:]
        
    def get_unaugmented_train_loader(self, batch_size: int = None, shuffle: bool = True):
        if batch_size is None:
            batch_size = self.train_batch_size
        if self.unique_label is None:
            loaders = [ds.get_unaugmented_train_loader for ds in self.nested_datasets]
            return LSUNData.merge_loaders_(loaders[0], loaders[1], batch_size, shuffle)
        else:
            return super().get_unaugmented_train_loader(batch_size, shuffle)
    
    def get_train_loader(self, batch_size: int = None, shuffle: bool = True):
        if batch_size is None:
            batch_size = self.train_batch_size
        if self.unique_label is None:
            loaders = [ds.get_train_loader for ds in self.nested_datasets]
            return LSUNData.merge_loaders_(loaders[0], loaders[1], batch_size, shuffle)
        else:
            return super().get_train_loader(batch_size, shuffle)
    
    def get_test_loader(self, batch_size: int = None, shuffle: bool = True):
        if batch_size is None:
            batch_size = self.test_batch_size
        if self.unique_label is None:
            loaders = [ds.get_test_loader for ds in self.nested_datasets]
            return LSUNData.merge_loaders_(loaders[0], loaders[1], batch_size, shuffle)
        else:
            return super().get_test_loader(batch_size, shuffle)
