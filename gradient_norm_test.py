import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import hydra
import wandb
import logging
from omegaconf import DictConfig, OmegaConf

from src.data.chexpert_data import CheXpertDataSet
from src.federated_learning.client import Client
from src.federated_learning.evaluation import evaluate_model
from src.trainer import Trainer
from src.custom_models import DenseNet, ResNet
from train_FL import get_dataset_transforms

BASE_PATH = './data/mendeley_test_data_partition'
MENDELEY_PATH = './data/'

@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    use_gpu = torch.cuda.is_available()
    random_seed = cfg.seed
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(random_seed)
        np.random.seed(random_seed)

    if cfg.debug:
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(message)s', level=logging.INFO)

    logging.info(OmegaConf.to_yaml(cfg, resolve=True))

    # ======== WandB Setup ========
    if cfg.wandb.disabled:
        wandb.init(mode='disabled')
    else:
        # Initialise WandB
        wandb_tags = ['gradient_norms', cfg.model.net, cfg.model.freeze_layers]
        wandb.init(project=(cfg.wandb.project or 'CXR_Reconstruction'),
                   entity="bjarnepfitzner",
                   group=cfg.wandb.group, name=cfg.wandb.name, tags=wandb_tags, resume='allow',
                   config=OmegaConf.to_container(cfg, resolve=True), allow_val_change=True,
                   settings=wandb.Settings(start_method="fork"))

    # Setup data and client
    train_transform_sequence, test_transform_sequence = get_dataset_transforms(cfg.data,
                                                                               cfg.model.input_layer_aggregation,
                                                                               cfg.model.pre_trained)

    single_client = Client('SingleClient')

    single_client.train_data = CheXpertDataSet(MENDELEY_PATH, f'{BASE_PATH}/train.csv', cfg.data.class_idx, cfg.data.policy,
                                 colour_input='L', transform=train_transform_sequence)
    single_client.train_loader = DataLoader(dataset=single_client.train_data, batch_size=cfg.training.B, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=False)

    single_client.val_data = CheXpertDataSet(MENDELEY_PATH, f'{BASE_PATH}/val.csv', cfg.data.class_idx, cfg.data.policy,
                               colour_input='L', transform=test_transform_sequence)
    single_client.val_loader = DataLoader(dataset=single_client.val_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    single_client.test_data = CheXpertDataSet(MENDELEY_PATH, f'{BASE_PATH}/test.csv', cfg.data.class_idx, cfg.data.policy,
                                colour_input='L', transform=test_transform_sequence)
    single_client.test_loader = DataLoader(dataset=single_client.test_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    if cfg.model.net == 'DenseNet121':
        net = DenseNet
    elif cfg.model.net.startswith('ResNet'):
        net = ResNet
    # create global model
    if use_gpu:
        global_model = net(version=cfg.model.net,
                           out_size=len(cfg.data.class_idx),
                           input_layer_aggregation=cfg.model.input_layer_aggregation,
                           pre_trained=cfg.model.pre_trained).cuda()
        # model=torch.nn.DataParallel(model).cuda()
    else:
        global_model = net(len(cfg.data.class_idx), cfg.model.input_layer_aggregation, cfg.model.pre_trained)

    single_client.model = global_model
    single_client.init_optimizer(cfg.training)
    single_client.output_path = './gradient_norm_out/'

    metrics_names = ['auroc', 'auprc', 'auprc_macroavg', 'f1', 'accuracy', 'mcc']
    header = ['epoch', 'time', 'train loss', 'val loss'] + metrics_names
    if cfg.training.track_norm:
        header += ['track_norm']

    all_grad_norms = []
    for epoch in range(cfg.training.T):
        all_metrics = Trainer.train(single_client, cfg, use_gpu, out_csv='gradient_norm_out.csv', freeze_mode=cfg.model.freeze_layers)
        #print(all_metrics)
        all_grad_norms.append(all_metrics[-1])
        wandb.log({
            'train/gradient_norm': wandb.Histogram(all_metrics[-1]),
            'train/gradient_norm_median': np.median(all_metrics[-1]),
            'train/gradient_norm_rolling_median': np.median(np.concatenate(all_grad_norms))
        }, step=epoch)
        evaluate_model(single_client.model, [single_client], epoch, 'val', use_gpu, cfg, plot_curves=False)
    wandb.log({'train/all_gradients': wandb.Table(columns=list(range(len(all_grad_norms[0]))),
                                                  data=all_grad_norms)},
              step=epoch)

    evaluate_model(single_client.model, [single_client], epoch, 'test', use_gpu, cfg, plot_curves=True)

if __name__ == "__main__":
    main()
