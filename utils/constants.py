
AGGREGATOR_TYPE = {
    "FedEdg": "EdgCentralized",
    "FedEM": "centralized",
    "FedAvg": "centralized",
    "FedProx": "centralized"
}

LOADER_TYPE = {
    "synthetic": "tabular",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "shakespeare": "shakespeare",
}

CLIENT_TYPE = {
    "FedEM": "mixture",
    "L2SGD": "normal",
    "FedAvg": "normal",
    "FedProx": "normal"
}

EXTENSIONS = {
    "tabular": ".pkl",
    "cifar10": ".pkl",
    "cifar100": ".pkl",
    "emnist": ".pkl",
    "femnist": ".pt",
    "shakespeare": ".txt",
}