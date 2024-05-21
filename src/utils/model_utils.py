import ast

from torch import nn

def freeze_batchnorm(model):

    """Modify model to not track gradients of batch norm layers
    and set them to eval() mode (no running stats updated)"""

    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()


def freeze_all_but_last(model):

    """Modify model to not track gradients of all but the last classification layer.
    Note: This is customized to the module naming of ResNet and DenseNet architectures."""

    for name, param in model.named_parameters():
        if 'fc' not in name and 'classifier' not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)


def freeze_middle(model):

    """Modify model to not track gradients of all but the first convolutional and last classification layer.
    Note: This is customized to the module naming of ResNet and DenseNet architectures above."""

    for name, param in model.named_parameters():
        if not any(part in name for part in ['fc', 'classifier', 'resnet50.conv1.weight', 'resnet18.conv1.weight',
                                             'densenet121.features.conv0.weight']):
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)


def median_grad_norm(path_csv, max_rounds=10, n_clients=1):

    """Compute the median L2 grad norm from grad norm lists
    saved in train_results.csv.
    Args:
        path_csv (str): Relative path to CSV with 'grad_norm' column containing parameter
        layer wise L2 grad norms per client per round.
        Grad norms are assumed to be a string representation of a list of values.

        max_rounds (int): Number of rounds to consider for median computation. If the number
        exceeds the total number of rounds, it will default to considering all.

        n_clients (int): Number of clients for which training was recorded.
    Returns:
        (array): Per parameter layer median gradient norms computed over clients over rounds.
        (float): Single median gradient norm over all gradient norm values."""

    df = pd.read_csv(path_csv)
    norms = np.array([ast.literal_eval(norms) for norms in df['track_norm']])
    if len(norms) > max_rounds:
        norms = norms[:max_rounds*n_clients] # keep first n rounds
    median_norms_params = np.median(norms, axis=0)
    median_norms_single = np.median(norms)

    return norms, median_norms_params, median_norms_single
