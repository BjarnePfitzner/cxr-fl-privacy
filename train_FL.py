"""Train a model using federated learning"""

#set which GPUs to use
import os
#selected_gpus = [7] #configure this
#os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in selected_gpus])

from PIL import Image
from datetime import datetime
import random
import numpy as np
import copy
import logging
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import opacus

from src.data.chexpert_data import CheXpertDataSet
from src.federated_learning.client import Client
from src.federated_learning.training import run_federated_learning
from src.federated_learning.evaluation import evaluate_model
from src.custom_models import DenseNet, ResNet
from src.utils.model_utils import freeze_all_but_last, freeze_batchnorm, freeze_middle
from src.utils.io_utils import check_path

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)

CHEXPERT_MEAN = [0.5029, 0.5029, 0.5029]
CHEXPERT_STD = [0.2899, 0.2899, 0.2899]

CHEXPERT_CLIENT_DATA = './data/chexpert_clients/'
CHEXPERT_PATH = '/dhc/dsets/ChestXrays/CheXpert/'
MENDELEY_CLIENT_DATA = './data/mendeley_clients/'
MENDELEY_PATH = './data/'

@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    use_gpu = torch.cuda.is_available()
    check_gpu_usage(use_gpu)

    # Check Paths
    #check_path(CHEXPERT_PATH, warn_exists=False, require_exists=True)
    #check_path(CHEXPERT_CLIENT_DATA, warn_exists=False, require_exists=True)
    #check_path(MENDELEY_PATH, warn_exists=False, require_exists=True)
    #check_path(MENDELEY_CLIENT_DATA, warn_exists=False, require_exists=True)

    #only use pytorch randomness for direct usage with pytorch
    random_seed = cfg.seed
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(random_seed)
        np.random.seed(random_seed)

    if cfg.debug:
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(message)s', level=logging.INFO)

    # ======== WandB Setup ========
    if cfg.wandb.disabled:
        logging.info('Disabled Wandb logging')
        wandb.init(mode='disabled')
        cfg.output_path += datetime.now().strftime("%Y-%m-%d/%H-%M-%S/")
    else:
        # Initialise WandB
        wandb_tags = [cfg.model.net, cfg.model.freeze_layers, f'{cfg.data.clients}_clients']
        if cfg.dp.enabled:
            wandb_tags.append('DP')
        wandb_run = wandb.init(project=(cfg.wandb.project or 'CXR_Reconstruction'),
                   entity="bjarnepfitzner",
                   group=cfg.wandb.group, name=cfg.wandb.name, tags=wandb_tags, resume='allow',
                   config=OmegaConf.to_container(cfg, resolve=True), allow_val_change=True,
                   settings=wandb.Settings(start_method="fork"))
        if wandb_run.sweep_id is not None:
            cfg.output_path += f'{wandb_run.sweep_id}/{wandb_run.id}'
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))

    freeze_dict = {'batch_norm': freeze_batchnorm, 'all_but_last': freeze_all_but_last, 'middle': freeze_middle}

    if cfg.model.net == 'DenseNet121':
        net = DenseNet
    elif cfg.model.net.startswith('ResNet'):
        net = ResNet
    else:
        raise NotImplementedError('Model type not implemented')

    # define client ids
    # ugly but okay
    if cfg.data.clients == 'all':
        chexpert_client_ids = list(range(22))
        mendeley_client_ids = list(range(18))
    elif cfg.data.clients == 'small':
        chexpert_client_ids = list(range(5, 22))
        mendeley_client_ids = list(range(18))
    elif cfg.data.clients == 'cxp':
        chexpert_client_ids = list(range(22))
        mendeley_client_ids = []
    elif cfg.data.clients == 'cxp_small':
        chexpert_client_ids = list(range(5, 22))
        mendeley_client_ids = []
    elif cfg.data.clients == 'cxp_large':
        chexpert_client_ids = list(range(5))
        mendeley_client_ids = []
    elif cfg.data.clients == 'mdl':
        chexpert_client_ids = []
        mendeley_client_ids = list(range(18))
    else:
        raise ValueError('Clients needs to be one of "all", "small", "cxp" "cxp_small", "cxp_large", "mdl"')

    # federated learning parameters
    num_clients = len(chexpert_client_ids) + len(mendeley_client_ids)
    if cfg.training.max_client_selections is not None:
        assert (num_clients * cfg.training.max_client_selections >=
                round(num_clients * cfg.training.q) * cfg.training.T), "Client fraction or maximum rounds for client selection is not large enough."

    if cfg.dp.enabled:
        assert cfg.model.freeze_layers == 'batch_norm' or cfg.model.freeze_layers == 'all_but_last', "Batch norm layers must be frozen for private training."

    # initialize client instances and their datasets
    clients = init_clients(cfg.data, cfg.training.B, cfg.model.input_layer_aggregation, cfg.dp.enabled,
                           cfg.model.pre_trained, chexpert_client_ids, mendeley_client_ids)

    # create global model
    if use_gpu:
        global_model = net(version=cfg.model.net,
                           out_size=len(cfg.data.class_idx),
                           input_layer_aggregation=cfg.model.input_layer_aggregation,
                           pre_trained=cfg.model.pre_trained).cuda()
        #global_model = torch.nn.DataParallel(global_model).cuda()
    else:
        global_model = net(len(cfg.data.class_idx), cfg.model.input_layer_aggregation, cfg.model.pre_trained)

    if cfg.model.checkpoint is not None: # load weights if some model is specified
        checkpoint = torch.load(cfg.model.checkpoint)
        if 'state_dict' in checkpoint:
            global_model.load_state_dict(checkpoint['state_dict'])
        else:
            global_model.load_state_dict(checkpoint)

    # freeze batch norm layers already, so it passes the privacy engine checks
    if cfg.model.freeze_layers != 'none':
         freeze_dict[cfg.model.freeze_layers](global_model)

    if cfg.save_model:
        # define path to store results in
        cfg.output_path = check_path(cfg.output_path, warn_exists=True)
        # save initial global model parameters
        torch.save({'state_dict': global_model.state_dict()}, cfg.output_path + 'global_init.pth.tar')

    # initialize client models and optimizers
    for client_k in clients:
        logging.debug(f"Initializing model and optimizer of {client_k.name}")
        client_k.model = copy.deepcopy(global_model)
        client_k.init_optimizer(cfg.training)

        if cfg.dp.enabled:
            # compute personal delta dependent on client's dataset size, or choose min delta value allowed
            client_k.delta = min(cfg.dp.max_delta, 1 / client_k.n_data * 0.1)
            logging.debug(f'Client delta: {client_k.delta}')
            # attach DP privacy engine for private training

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                client_k.privacy_engine = opacus.PrivacyEngine(client_k.model,
                                                               target_epsilon=cfg.dp.epsilon,
                                                               target_delta=client_k.delta,
                                                               max_grad_norm=cfg.dp.S,
                                                               epochs=cfg.training.max_client_selections,
                                                               # noise_multiplier=cfg.dp.z,
                                                               # get sample rate with respect to client's dataset
                                                               sample_rate=min(1, cfg.training.B / client_k.n_data))
            client_k.privacy_engine.attach(client_k.optimizer)

    # =========== DO FEDERATED LEARNING =============
    trained_model = run_federated_learning(global_model, clients, use_gpu, cfg)

    # ======= Test final model =========
    evaluate_model(trained_model, clients, cfg.training.T, 'test', use_gpu, cfg, plot_curves=True)

    wandb.finish()


def init_clients(data_cfg, batch_size, input_layer_aggregation, dp_training, pre_trained, chexpert_client_ids, mendeley_client_ids):
    if input_layer_aggregation == 'repeat':
        colour_input = 'RGB'
    else:
        colour_input = 'L'
    train_transform_sequence, test_transform_sequence = get_dataset_transforms(data_cfg, input_layer_aggregation, pre_trained)
    clients = ([Client(name=f'CXP_client_{n}') for n in chexpert_client_ids] +
               [Client(name=f'MDL_client_{n}') for n in mendeley_client_ids])
    num_clients = len(chexpert_client_ids) + len(mendeley_client_ids)

    for i in range(num_clients):
        cur_client = clients[i]
        logging.debug(f"Initializing {cur_client.name}")

        if 'CXP' in cur_client.name:
            data_path = CHEXPERT_PATH
            path_to_client = check_path(CHEXPERT_CLIENT_DATA + cur_client.name.replace('CXP_', ''),
                                        warn_exists=False, require_exists=True)
        else:
            data_path = MENDELEY_PATH
            path_to_client = check_path(MENDELEY_CLIENT_DATA + cur_client.name.replace('MDL_', ''),
                                        warn_exists=False, require_exists=True)

        train_file = path_to_client + '/client_train.csv'
        cur_client.train_data = CheXpertDataSet(data_path, train_file, data_cfg.class_idx, data_cfg.policy,
                                                colour_input=colour_input, transform=train_transform_sequence)

        assert cur_client.train_data[0][0].shape == torch.Size([len(colour_input), data_cfg.resize, data_cfg.resize])
        assert cur_client.train_data[0][1].shape == torch.Size([len(data_cfg.class_idx)])

        cur_client.n_data = len(cur_client.train_data)
        logging.debug(f"Holds {cur_client.n_data} data points")

        # for LDP, drop last incomplete batch if dataset has at least one full batch, otherwise keep one incomplete batch
        if dp_training and cur_client.n_data > batch_size:
            drop_last = True
            logging.debug(f"Dropping incomplete batch of {cur_client.n_data % batch_size} data points")
            cur_client.n_data = cur_client.n_data - cur_client.n_data % batch_size
        else:
            drop_last = False

        cur_client.train_loader = DataLoader(dataset=cur_client.train_data, batch_size=batch_size, shuffle=True,
                                             num_workers=8, persistent_workers=True, pin_memory=True,
                                             drop_last=drop_last)

        val_file = path_to_client + 'client_val.csv'
        test_file = path_to_client + 'client_test.csv'

        if os.path.exists(val_file):
            cur_client.val_data = CheXpertDataSet(data_path, val_file, data_cfg.class_idx, data_cfg.policy,
                                                  colour_input=colour_input, transform=test_transform_sequence)
            cur_client.test_data = CheXpertDataSet(data_path, test_file, data_cfg.class_idx, data_cfg.policy,
                                                   colour_input=colour_input, transform=test_transform_sequence)

            cur_client.val_loader = DataLoader(dataset=cur_client.val_data, batch_size=128, shuffle=False,
                                               num_workers=8, persistent_workers=True, pin_memory=True)
            cur_client.test_loader = DataLoader(dataset=cur_client.test_data, batch_size=128, shuffle=False,
                                                num_workers=8, persistent_workers=True, pin_memory=True)

        else: # clients that don't
            logging.debug(f"No validation data for client{i}")
            cur_client.val_loader = None
            cur_client.test_loader = None

    return clients


def get_dataset_transforms(dataset_cfg: DictConfig, input_layer_aggregation, pre_trained: bool):
    #define mean and std dependent on whether using a pretrained model
    if pre_trained:
        data_mean = IMAGENET_MEAN
        data_std = IMAGENET_STD
    else:
        data_mean = CHEXPERT_MEAN
        data_std = CHEXPERT_STD

    if input_layer_aggregation == 'mean':
        data_mean = np.mean(data_mean)
        data_std = np.mean(data_std)
    elif input_layer_aggregation == 'sum':
        data_mean = np.sum(data_mean)
        data_std = np.sum(data_std)

    # define transforms
    # if using augmentation, use different transforms for training, test & val data
    if dataset_cfg.augment:
        train_transform_sequence = transforms.Compose([transforms.Resize((dataset_cfg.resize, dataset_cfg.resize)),
                                                # transforms.RandomResizedCrop(imgtransResize),
                                                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(data_mean, data_std)
                                                ])
    else:
        #no augmentation for comparison with DP
        train_transform_sequence = transforms.Compose([transforms.Resize((dataset_cfg.resize, dataset_cfg.resize)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(data_mean, data_std)
                                                ])

    test_transform_sequence = transforms.Compose([transforms.Resize((dataset_cfg.resize, dataset_cfg.resize)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(data_mean, data_std)
                                            ])

    return train_transform_sequence, test_transform_sequence


def check_gpu_usage(use_gpu):

    """Give feedback to whether GPU is available and if the expected number of GPUs are visible to PyTorch.
    """
    assert use_gpu is True, "GPU not used"
    #assert torch.cuda.device_count() == len(selected_gpus), "Wrong number of GPUs available to Pytorch"
    logging.info(f"{torch.cuda.device_count()} GPUs available")

    return True




if __name__ == "__main__":
    main()
