import torch
import torch.utils.data
from torchvision import datasets, transforms
from functools import partial

DEFAULT_OPTS = {
    'batch_size': 32,
    'shuffle': True,
    'drop_last': True
}

DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.ToTensor()
])

def get_torchvision_loaders(dataset: str, save_loc='.', loader_opts=DEFAULT_OPTS, transforms=DEFAULT_TRANSFORMS):
    """
    Helper to prepare train and test data loaders.
    :param dataset: name of any downloadable torch vision dataset:
        'MNIST', 'FashionMNIST', 'EMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'SVHN'
    :param save_loc:
    :param loader_opts:
    :param transforms: optional data transforms
    :return:
    """

    # pre-load common arguments
    DataClass = getattr(datasets, dataset)
    DataLoader = partial(torch.utils.data.DataLoader, **loader_opts)

    # create / download datasets
    train_dataset = DataClass(save_loc, train=True, download=True, transform=transforms)
    test_dataset = DataClass(save_loc, train=False, transform=transforms)

    # wrap with PyTorch DataLoader
    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)

    return train_loader, test_loader