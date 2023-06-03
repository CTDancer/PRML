import torch
from torchvision import datasets, transforms

def get_dataset(dataset):

    dst_train_dict = None
    class_map = None
    loader_train_dict = None
    class_map_inv = None

    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        data_folder = r'c:\Users\Life_Dancer\Desktop\PRML\assignment2\data\cifar'

        dst_train = datasets.CIFAR10(data_folder, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_folder, train=False, download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=8, shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(dst_test, batch_size=8, shuffle=False, num_workers=2)

    elif dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.5]
        std = [0.5]

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        data_folder = r'c:\Users\Life_Dancer\Desktop\PRML\assignment2\data\mnist'

        dst_train = datasets.MNIST(data_folder, train=True, download=True, transform=transform)
        dst_test = datasets.MNIST(data_folder, train=False, download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=8, shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(dst_test, batch_size=8, shuffle=False, num_workers=2)

    else:
        exit('unknown dataset: %s' % dataset)

    return trainloader, testloader
