import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


from cs236781.train_results import FitResult

from .cnn import CNN, ResNet
from .mlp import MLP
from .training import ClassifierTrainer
from .classifier import ArgMaxClassifier, BinaryClassifier, select_roc_thresh

DATA_DIR = os.path.expanduser("~/.pytorch-datasets")

MODEL_TYPES = {
    ###
    "cnn": CNN,
    "resnet": ResNet,
}

OPTIMIZERS = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
LOSSES = {"cross entropy":torch.nn.CrossEntropyLoss}



def mlp_experiment(
    depth: int,
    width: int,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    dl_test: DataLoader,
    n_epochs: int,
):
    # TODO:
    #  - Create a BinaryClassifier model.
    #  - Train using our ClassifierTrainer for n_epochs, while validating on the
    #    validation set.
    #  - Use the validation set for threshold selection.
    #  - Set optimal threshold and evaluate one epoch on the test set.
    #  - Return the model, the optimal threshold value, the accuracy on the validation
    #    set (from the last epoch) and the accuracy on the test set (from a single
    #    epoch).
    #  Note: use print_every=0, verbose=False, plot=False where relevant to prevent
    #  output from this function.
    # ====== YOUR CODE: ======
    hp_arch = {
        'n_layers': depth,  # Number of hidden layers
        'hidden_dims': width,  # Number of neurons in each hidden layer
        'activation': torch.nn.LeakyReLU(0.05),  # Activation function for hidden layers
        'out_activation': 'softmax'  # Output layer activation function
    }

    hp_optim = {
        'lr': 0.001,
        'weight_decay': 0.001,
        'betas': (0.7, 0.99),
        'loss_fn': torch.nn.CrossEntropyLoss()  # Loss function for training
    }

    model = BinaryClassifier(
        model=MLP(
            in_dim=2,
            dims=[hp_arch['hidden_dims']] * hp_arch['n_layers'] + [2],
            nonlins=[hp_arch['activation']] * hp_arch['n_layers'] + [hp_arch['out_activation']]
        ),
        threshold=0.5,
    )

    loss_fn = hp_optim.pop('loss_fn')
    optimizer = torch.optim.Adam(model.parameters(), **hp_optim)
    trainer = ClassifierTrainer(model, loss_fn, optimizer)
    
    valid_acc = trainer.fit(dl_train, dl_valid, num_epochs=n_epochs, print_every=0, verbose=False, early_stopping=3).test_acc[-1]
    thresh = select_roc_thresh(model, *dl_valid.dataset.tensors, plot=False)
    model.threshold = thresh
    test_acc = trainer.test_epoch(dl_test).accuracy
    # ========================
    return model, thresh, valid_acc, test_acc
    
    

def cnn_experiment(
    run_name,
    out_dir="./results",
    seed=None,
    device=None,
    # Training params
    bs_train=128,
    bs_test=None,
    batches=100,
    epochs=100,
    early_stopping=10,
    checkpoints=None,
    lr=1e-3,
    reg=1e-3,
    # Model params
    filters_per_layer=[64],
    layers_per_block=2,
    pool_every=1,
    hidden_dims=[1024],
    model_type="cnn",
    # You can add extra configuration for your experiments here
    activation_type = "relu",
    activation_params =dict(),
    pooling_type = "max",
    pooling_params= dict(kernel_size=2),
    batchnorm=True,
    dropout=0.163,
    bottleneck=False,
    loss_fn = "cross entropy",
    optimizer = "Adam",
    hp_optim = dict(betas=(0.996, 0.983)),
    subset=False
):
    """
    Executes a single run of a Part3 experiment with a single configuration.
    These parameters are populated by the CLI parser below.
    See the help string of each parameter for it's meaning.
    """

    # TODO: implement batchs 
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ])
    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)
    if subset:
        train_idx = list(range(0, subset))
        test_idx = list(range(0, subset//4))
        ds_train = torch.utils.data.Subset(ds_train, train_idx)
        ds_test = torch.utils.data.Subset(ds_test, test_idx)
    print(f"dataset lengths: {len(ds_train)}, {len(ds_test)}", flush=True)

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]

    # TODO: Train
    #  - Create model, loss, optimizer and trainer based on the parameters.
    #    Use the model you've implemented previously, cross entropy loss and
    #    any optimizer that you wish.
    #  - Run training and save the FitResults in the fit_res variable.
    #  - The fit results and all the experiment parameters will then be saved
    #   for you automatically.
    fit_res = None
    # ====== YOUR CODE: 
    L = layers_per_block
    K = filters_per_layer
    
    dl_train = DataLoader(ds_train, batch_size=bs_train)
    dl_test = DataLoader(ds_test, batch_size=bs_test)
    sample_shape = next(iter(dl_train))[0][0].shape
    conv_channels = [elem for elem, count in zip(K, [L]*len(K)) for i in range(count)]
    net_params = dict(
        in_size=sample_shape, out_classes=10, channels=conv_channels,
        pool_every=pool_every, hidden_dims=hidden_dims,
        activation_type=activation_type, activation_params=activation_params,
        pooling_type=pooling_type, pooling_params=pooling_params,
        batchnorm=batchnorm, dropout=dropout,
        bottleneck=bottleneck
    )
    if model_type == 'cnn':
        for key in ['batchnorm', 'dropout', 'bottleneck']:
            net_params.pop(key, None)
        net_params['conv_params']=dict(kernel_size=3, stride=1, padding=1)
    # print(net_params)
    model = ArgMaxClassifier(
    model=model_cls(**net_params)
    )
    
    print(cfg)
    optimizer = OPTIMIZERS[optimizer](params=model.parameters(),lr=lr, weight_decay=reg, **hp_optim)
    trainer = ClassifierTrainer(model, LOSSES[loss_fn](), optimizer, device)
    fit_res = trainer.fit(dl_train, dl_test, num_epochs=epochs, print_every=1, verbose=False, early_stopping=early_stopping, checkpoints=checkpoints)
    # ========================

    save_experiment(run_name, out_dir, cfg, fit_res)


def define_model(trial, layers_per_block, filters_per_layer, model_type='resnet'):
    L = layers_per_block
    K = filters_per_layer
    conv_channels = [elem for elem, count in zip(K, [L]*len(K)) for i in range(count)]
    dropout = trial.suggest_float('dropout', 0.1,0.3)
    pool_every = trial.suggest_int('pool_every', layers_per_block//4,6)
    hidden_dims_val = trial.suggest_int("hidden_dims_val", 256,1024,256)
    hidden_dims_num = trial.suggest_int("hidden_dims_num", 1,4)
    hidden_dims = [ hidden_dims_val]* hidden_dims_num
    net_params = dict(
        in_size=[3,32,32], out_classes=10, channels=conv_channels,
        pool_every=pool_every, hidden_dims=hidden_dims,
        activation_type='lrelu', activation_params=dict(negative_slope=0.05),
        pooling_type='max', pooling_params=dict(kernel_size=2),
        batchnorm=True, dropout=dropout,
        bottleneck=False
    )
    # print(net_params)
    if model_type == 'cnn':
        for key in ['batchnorm', 'dropout', 'bottleneck']:
            net_params.pop(key, None)
        net_params['conv_params']=dict(kernel_size=3, stride=1, padding=1)
    model = ArgMaxClassifier(
    model=MODEL_TYPES[model_type](**net_params)
    )
    return model

def objective(trial,run_name, layers_per_block, filters_per_layer, bs_train=128,
    bs_test=32, subset=0, model_type='resnet', optimizer='Adam'):
    model = define_model(trial, layers_per_block, filters_per_layer, model_type=model_type)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ]
)
    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)
    if subset:
        train_idx = list(range(0, subset))
        test_idx = list(range(0, subset//4))
        ds_train = torch.utils.data.Subset(ds_train, train_idx)
        ds_test = torch.utils.data.Subset(ds_test, test_idx)
    print(f"train size: {len(ds_train)}", f"test size: {len(ds_test)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_train = DataLoader(ds_train, batch_size=bs_train)
    dl_test = DataLoader(ds_test, batch_size=bs_test)
    print(dl_train.batch_sampler, dl_test.batch_sampler)
    
    weight_decay = trial.suggest_float('weight_decay', 1e-5,1e-2)
    if optimizer=='Adam':
        lr = trial.suggest_float('lr', 1e-5,1e-2)
        beta1 = trial.suggest_float('beta1', 0.7,1)
        beta2 = trial.suggest_float('beta2', 0.7,1)
        optimizer = OPTIMIZERS[optimizer](params=model.parameters(),lr=lr, weight_decay=weight_decay, betas=(beta1,beta2))
    elif optimizer == 'SGD':
        lr = trial.suggest_float('lr', 1e-4,1e-1)
        momentum = trial.suggest_float('momentum',  1e-4,1e-1)
        optimizer = OPTIMIZERS['SGD'](params=model.parameters(),lr=lr, weight_decay=weight_decay, momentum=momentum)
    trainer = ClassifierTrainer(model, LOSSES['cross entropy'](), optimizer, device)
    fit_res = trainer.fit(dl_train, dl_test, num_epochs=10, print_every=5, verbose=False, early_stopping=3, trial=trial)
    return fit_res.test_loss[-1]


def run_optuna_experiment(run_name, filters_per_layer, layers_per_block, subset=0, n_trials=50, out_dir="./results"):
    from optuna.trial import TrialState
    import optuna
    from typing import List, NamedTuple
    from cs236781.train_results import FitResult

    try:
        study = optuna.load_study(study_name=run_name, storage=f'sqlite:///{out_dir}/{run_name}.db')
    except KeyError:
        study = optuna.create_study(study_name=run_name, storage=f'sqlite:///{out_dir}/{run_name}.db')
    model_type = run_name.split("_")[1]
    optimizer = run_name.split("_")[-1]
    print(model_type, optimizer)
    study.optimize(lambda trial: objective(trial, run_name=run_name, filters_per_layer=filters_per_layer,
                                            layers_per_block=layers_per_block, subset=subset, model_type=model_type, optimizer=optimizer), n_trials=n_trials)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    fit = FitResult(10, [], [], [study.best_value], [])
    cfg = study.best_params
    cfg["layers_per_block"] = layers_per_block
    cfg["filters_per_layer"] = filters_per_layer
    save_experiment(run_name, "./results", cfg, fit)


def save_experiment(run_name, out_dir, cfg, fit_res):
    output = dict(config=cfg, results=fit_res._asdict())

    cfg_LK = (
        f'L{cfg["layers_per_block"]}_K'
        f'{"-".join(map(str, cfg["filters_per_layer"]))}'
    )
    output_filename = f"{os.path.join(out_dir, run_name)}_{cfg_LK}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"*** Output file {output_filename} written")


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)
    config = output["config"]
    fit_res = FitResult(**output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="CS236781 HW2 Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    sp_exp.set_defaults(subcmd_fn=cnn_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./results",
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )

    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=int,
        help="Save model checkpoints to this file when test " "accuracy improves",
        default=None,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    sp_exp.add_argument("--reg", type=float, help="L2 regularization", default=1e-3)

    # # Model
    sp_exp.add_argument(
        "--filters-per-layer",
        "-K",
        type=int,
        nargs="+",
        help="Number of filters per conv layer in a block",
        metavar="K",
        required=True,
    )
    sp_exp.add_argument(
        "--layers-per-block",
        "-L",
        type=int,
        metavar="L",
        help="Number of layers in each block",
        required=True,
    )
    sp_exp.add_argument(
        "--pool-every",
        "-P",
        type=int,
        metavar="P",
        help="Pool after this number of conv layers",
        required=True,
    )
    sp_exp.add_argument(
        "--hidden-dims",
        "-H",
        type=int,
        nargs="+",
        help="Output size of hidden linear layers",
        metavar="H",
        required=True,
    )
    sp_exp.add_argument(
        "--model-type",
        "-M",
        choices=MODEL_TYPES.keys(),
        default="cnn",
        help="Which model instance to create",
    )

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed



if __name__ == "__main__":
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))
