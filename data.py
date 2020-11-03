import os

import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_CIFAR(args):
    data_root_dir = args.data
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    train_dataset = datasets.CIFAR10(
        data_root_dir, train=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        download=False)

    val_dataset = datasets.CIFAR10(
        data_root_dir, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        download=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset)
    else:
        train_sampler, val_sampler = None, None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader, train_sampler
