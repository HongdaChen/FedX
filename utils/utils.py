import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aggregator import CentralizedAggregator
from client import Client
from datasets import get_cifar10, get_cifar100, TabularDataset, SubCIFAR10, SubCIFAR100
from learners.learner import Learner
from learners.learners_ensemble import LanguageModelingLearnersEnsemble, LearnersEnsemble
from models import get_mobilenet
from utils.constants import EXTENSIONS, LOADER_TYPE
from utils.metrics import accuracy
from utils.optim import get_optimizer, get_lr_scheduler



def get_aggregator(aggregator_type,
                   clients,
                   global_learners_ensemble,
                   lr,
                   lr_lambda,
                   mu,
                   communication_probability,
                   q,
                   sampling_rate,
                   log_freq,
                   global_train_logger,
                   global_test_logger,
                   test_clients,
                   verbose,
                   seed=None):
    """
    `personalized` corresponds to pFedMe
    :param aggregator_type:
    :param clients:
    :param global_learners_ensemble:
    :param lr: oly used with FLL aggregator
    :param lr_lambda: only used with Agnostic aggregator
    :param mu: penalization term, only used with L2SGD
    :param communication_probability: communication probability, only used with L2SGD
    :param q: fairness hyper-parameter, ony used for FFL client
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients:
    :param verbose: level of verbosity
    :param seed: default is None
    :return:
    """
    seed = (seed if (seed is not None and seed >=0) else int(time.time()))

    if aggregator_type == 'centralized':
        return CentralizedAggregator(clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed)
    # elif aggregator_type == 'L2SGD':
    #     return LoopLessLocalSGDAggregator()
    # elif aggregator_type == 'decentralized':
    #     n_clients = len(clients)
    #     mixing_matrix = get_mixing_matrix(n=n_clients,p=0.5,seed=seed)
    #     return DecentralizedAggregator()
    else:
        raise NotImplementedError(
            f"{aggregator_type} is not a possible aggregator type."
            " Available are: `centralized`, `L2SGD`,`decentralized`,"
        )

def get_learner(name,
                device,
                optimizer_name,
                scheduler_name,
                initial_lr,
                mu,
                n_rounds,
                seed,
                input_dim=None,
                output_dim=None):
    """
    constructs the learner corresponding to an experiment for a given seed
    :param name: name of the experiment to be used; possible are
                {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :return:
    """
    torch.manual_seed(seed)
    if name == "cifar10":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet(n_classes=10).to(device)
        # model = get_resnet18(n_classes=10).to(device)
        is_binary_classification = False
    elif name == "cifar100":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = get_mobilenet(n_classes=100).to(device)
        is_binary_classification = False
    else:
        raise NotImplementedError

    optimizer = get_optimizer(optimizer_name=optimizer_name,
                              model=model,
                              lr_initial=initial_lr,
                              mu=mu)
    lr_scheduler = get_lr_scheduler(optimizer=optimizer,
                                    scheduler_name=scheduler_name,
                                    n_rounds=n_rounds)
    if name == 'shakespeare':
        pass
    else:
        return Learner(model=model,
                       criterion=criterion,
                       metric=metric,
                       device=device,
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler,
                       is_binary_classification=is_binary_classification)


def get_learners_ensemble(n_learners,
                          name,
                          device,
                          optimizer_name,
                          scheduler_name,
                          initial_lr,
                          mu,
                          n_rounds,
                          seed,
                          input_dim=None,
                          output_dim=None):
    """
    only add a parameter `n_learners` compare with get_learner()
    constructs the learner corresponding to an experiment for a given seed
    :param n_learners: number of learners in the ensemble
    :param name: name of the experiment to be used; possible are
                {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :return:
    """
    learners = [
        get_learner(
            name=name,
            device=device,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            initial_lr=initial_lr,
            input_dim=input_dim,
            output_dim=output_dim,
            n_rounds=n_rounds,
            seed=seed + learner_id,
            mu=mu
        ) for learner_id in range(n_learners)
    ]
    learners_weights = torch.ones(n_learners)/n_learners
    if name == "shakespeare":
        return LanguageModelingLearnersEnsemble(learners=learners, learners_weights=learners_weights)
    else:
        return LearnersEnsemble(learners=learners, learners_weights=learners_weights)


def get_client(client_type,
        learners_ensemble,
        q,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        tune_locally):
    """
    :param client_type:
    :param learners_ensemble:
    :param q: fairness hyper-parameter, ony used for FFL client
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param tune_locally
    :return:
    """
    if client_type == "mixture":
        pass
    else:
        return Client(learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally)

def get_loader(type_, path, batch_size, train, inputs=None, targets=None):
    """
    constructs a torch.utils.DataLoader object from the given path
    :param type_: type of the dataset; possible are `tabular`, `images` and `text`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :return: torch.utils.DataLoader
    """
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "cifar10":
        dataset = SubCIFAR10(path, cifar10_data=inputs, cifar10_targets=targets)
    elif type_ == "cifar100":
        dataset = SubCIFAR100(path, cifar100_data=inputs, cifar100_targets=targets)
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

        # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = ((type_ == "cifar100") or (type_ == "cifar10")) and (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)


def get_loaders(type_,root_path,batch_size,is_validation):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last
    :param type_: type of the dataset;
    :param root_path: path to the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])
    """

    if type_ == "cifar10":
        inputs, targets = get_cifar10()
    elif type_ == "cifar100":
        inputs, targets = get_cifar100()
    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []

    for task_id, task_dir in enumerate(tqdm(os.listdir(root_path))):
        task_data_path = os.path.join(root_path, task_dir)

        train_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=True
            )

        val_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)

    return train_iterators, val_iterators, test_iterators
