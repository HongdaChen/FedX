import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.constants import AGGREGATOR_TYPE, LOADER_TYPE, CLIENT_TYPE
from utils.args import concat_logs_path
from utils.args import parse_args
from utils.utils import get_aggregator, get_learners_ensemble, get_client, get_loaders


def init_clients(args_, root_path, logs_dir):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("=======> Building data iterators...")

    train_iterators, val_iterators, test_iterators = get_loaders(
        type_=LOADER_TYPE[args_.dataset],
        root_path=root_path,
        batch_size=args_.bz,
        is_validation=args_.validation
    )
    print("=======> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators,val_iterators,test_iterators), total=len(train_iterators))):
        if train_iterator is None or test_iterator is None:
            continue
        learners_ensemble = get_learners_ensemble(n_learners=args_.n_learners,
                name=args_.dataset,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu)
        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path,exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(client_type=CLIENT_TYPE[args_.algorithm],
                            learners_ensemble=learners_ensemble,
                            q=args_.q,
                            train_iterator=train_iterator,
                            val_iterator=val_iterator,
                            test_iterator=test_iterator,
                            logger=logger,
                            local_steps=args_.local_steps,
                            tune_locally=args_.locally_tune_clients)
        clients_.append(client)
    return clients_

def run_experiment(args_):
    # set random seed to ensure the result can be repeated
    torch.manual_seed(args_.seed)
    torch.cuda.manual_seed(args_.seed)
    torch.cuda.manual_seed_all(args_.seed)

    ############################## Logs ########################################
    # tell the client where the seperated data is from
    data_dir = os.path.join("data",args_.dataset,"all_data")
    # set the tensorboard logs dir
    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", concat_logs_path(args_))
    # set and make logs dir
    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    ############################## Clients ########################################
    # Client initialization: tell each client to get the data belonging to itself
    print("===> Client initialization...")
    clients = init_clients(args_,
                           root_path=os.path.join(data_dir, "train"),
                           logs_dir=os.path.join(logs_dir, "train"))
    # Test Clients initialization: not necessary
    print("===> Test Clients initialization...")
    test_clients = init_clients(args_,
                                root_path=os.path.join(data_dir, "test"),
                                logs_dir=os.path.join(logs_dir, "test"))
    ############################## Learners #######################################
    # ensemble learners
    global_learners_ensemble = get_learners_ensemble(n_learners=args_.n_learners,
                                                     name=args_.dataset,
                                                     device=args_.device,
                                                     optimizer_name=args_.optimizer,
                                                     scheduler_name=args_.lr_scheduler,
                                                     initial_lr=args_.lr,
                                                     input_dim=args_.input_dimension,
                                                     output_dim=args_.output_dimension,
                                                     n_rounds=args_.n_rounds,
                                                     seed=args_.seed,
                                                     mu=args_.mu)


    ############################## Aggregator #####################################
    if args_.decentralized:
        aggregator_type = "decentralized"
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.algorithm]

    aggregator = get_aggregator(aggregator_type=aggregator_type,
                                clients=clients,
                                global_learners_ensemble=global_learners_ensemble,
                                lr_lambda=args_.lr_lambda,
                                lr=args_.lr,
                                q=args_.q,
                                mu=args_.mu,
                                communication_probability=args_.communication_probability,
                                sampling_rate=args_.sampling_rate,
                                log_freq=args_.log_freq,
                                global_train_logger=global_train_logger,
                                global_test_logger=global_test_logger,
                                test_clients=test_clients,
                                verbose=args_.verbose,
                                seed=args_.seed)


    ############################## Training #######################################
    print("Training...")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    while current_round <= args_.n_rounds:
        # parameter server executes
        aggregator.mix()
        # update the tqdm bar
        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round
    ############################## Saving #######################################
    if "save_dir" in args_:
        save_dir = os.path.join(args_.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        aggregator.save_state(save_dir)

if __name__ == "__main__":
    # similar to random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benmark = False

    args = parse_args()
    run_experiment(args)
