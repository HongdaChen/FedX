import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
    "Train/Loss",
    "Train/Metric",
    "Test/Loss",
    "Test/Metric"
]

FILES_NAMES = {
    "Train/Loss": "train-loss.png",
    "Train/Metric": "train-acc.png",
    "Test/Loss": "test-loss.png",
    "Test/Metric": "test-acc.png",
}

AXE_LABELS = {
    "Train/Loss": "Train loss",
    "Train/Metric": "Train acc",
    "Test/Loss": "Test loss",
    "Test/Metric": "Test acc",
}

LEGEND = {
    "FedAvg": "FedAvg",
    "FedEM": "FedEM",
    "FedEdg": "FedEdg",
    "FedAvg_adapt": "FedAvg+",
    "FedProx": "FedProx"
}

MARKERS = {
    "local": "x",
    "clustered": "s",
    "FedAvg": "h",
    "FedEM": "d",
    "FedEdg": "*",
    "FedAvg_adapt": "4",
    "personalized": "X",
    "DEM": "|",
    "FedProx": "p"
}

COLORS = {
    "local": "tab:blue",
    "clustered": "tab:orange",
    "FedAvg": "tab:green",
    "FedEM": "tab:red",
    "FedEdg": "tab:purple",
    "FedAvg_adapt": "tab:c",
    "personalized": "tab:brown",
    "DEM": "tab:pink",
    "FedProx": "tab:cyan"
}

def make_plot(path_,tag_,save_path=None):
    """

    :param path_: path of the logs directory, `path_` should contain sub-directories corresponding to algorithms
        each sub-directory must contain a single tf events file.
    :param tag_:
    :param save_path: default to the path_
    :return:
    """
    fig, ax = plt.subplots(figsize=(24,20))
    algorithms = [i for i in os.listdir(path_) if not i.endswith(".png")]
    for algorithm in algorithms:
        for mode in ["train"]:
            algorithm_path = os.path.join(path_,algorithm,mode)
            for task in os.listdir(algorithm_path):
                if task == "global":
                    task_path = os.path.join(algorithm_path,task)
                    ea = EventAccumulator(task_path).Reload()

                    tag_values = []
                    steps = []
                    for event in ea.Scalars(tag_):
                        tag_values.append(event.value)
                        steps.append(event.step)
                    if algorithm in LEGEND:
                        ax.plot(steps,tag_values,
                                linewidth=5.0,
                                marker=MARKERS[algorithm],
                                markersize=20,
                                markeredgewidth=5,
                                label=f"{LEGEND[algorithm]}",
                                color=COLORS[algorithm])
    ax.grid(True,linewidth=2)
    ax.set_ylabel(AXE_LABELS[tag_],fontsize=50)
    ax.set_xlabel("Rounds",fontsize=50)

    ax.tick_params(axis='both',labelsize=25)
    ax.legend(fontsize=60)

    # os.makedirs(save_path,exist_ok=True)
    fig_path = os.path.join(path_,f"{FILES_NAMES[tag_]}")
    plt.savefig(fig_path,bbox_inches='tight')

if __name__ == "__main__":
    make_plot("../logs/shakespeare", "Test/Metric")
    make_plot("../logs/shakespeare", "Test/Loss")
    make_plot("../logs/shakespeare", "Train/Metric")
    make_plot("../logs/shakespeare", "Train/Loss")




