import argparse
import collections
import math

import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model_PWR as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import matplotlib.pyplot as plt
import numpy as np
import random


# Helper function to show a batch
# from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
def show_batch(sample_batched, config):
    """Show image with labelmap overlay for a batch of samples."""
    image_batch, label_batch = sample_batched['image'], sample_batched['label']

    #TODO: convert sampled batch to image for show
    
    plt.figure()
    plt.imshow(i)
    plt.figure()
    plt.imshow(hm)
    plt.axis('off')
    plt.ioff()
    plt.show()


def test_dataloader(config):
    # logger = config.get_logger('train')
    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)

    for batch_idx, sample_batched in enumerate(data_loader):
        print('Batch id: ', batch_idx)
        show_batch(sample_batched, config)
        break


def test_model_mvlm(config):
    logger = config.get_logger('train')

    model = config.initialize('arch', module_arch)
    logger.info(model)


def get_cuda_info():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    print('Selected cuda device: ', torch.cuda.current_device())

    print('Number of GPUs available: ', torch.cuda.device_count())

    # Additional Info when using cuda
    if device.type == 'cuda':
        print('Cuda device name: ', torch.cuda.get_device_name(0))
        print('Cuda capabilities: ', torch.cuda.get_device_capability(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
        print('Max allocated:   ', round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1), 'GB')


def main(config):
    # logger = config.get_logger('train')

    # setup data_loader instances
    print('Initialising data loader')
    data_loader = config.initialize('data_loader', module_data)
    print('Initialising validation data')
    valid_data_loader = data_loader.split_validation()

    print('Initialising model')
    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    # logger.info(model)

    print('Initialising loss')
    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    print('Initialising optimizer')
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    print('Initialising scheduler')
    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    print('Initialising trainer')
    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    print('starting to train')
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    global_config = ConfigParser(args, options)
    main(global_config)
    # test_dataloader(global_config)
    # test_model_mvlm(config)
    # get_cuda_info(config)
