"""Train a model using federated learning"""

#set which GPUs to use
import os
#selected_gpus = [7] #configure this
#os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in selected_gpus])

import pandas as pd
import argparse
import json
from PIL import Image
import time
import random
import numpy as np
import csv
import copy

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import opacus

#local imports
from chexpert_data import CheXpertDataSet
from trainer import Trainer, DenseNet121, ResNet50, Client, freeze_batchnorm, freeze_all_but_last, freeze_middle
from utils import check_path, merge_eval_csv


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)

CHEXPERT_MEAN = [0.5029, 0.5029, 0.5029]
CHEXPERT_STD = [0.2899, 0.2899, 0.2899]

CHEXPERT_CLIENT_DATA = 'data/chexpert_clients'
CHEXPERT_PATH = '/dhc/dsets/ChestXrays/CheXpert/'
MENDELEY_CLIENT_DATA = 'data/chexpert_clients'
MENDELEY_PATH = './data/mendeley_xray/'

@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    use_gpu = torch.cuda.is_available()
    check_gpu_usage(use_gpu)

    # Check Paths
    check_path(CHEXPERT_PATH, warn_exists=False, require_exists=True)
    check_path(CHEXPERT_CLIENT_DATA, warn_exists=False, require_exists=True)
    check_path(MENDELEY_PATH, warn_exists=False, require_exists=True)
    check_path(MENDELEY_CLIENT_DATA, warn_exists=False, require_exists=True)

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

    freeze_dict = {'batch_norm': freeze_batchnorm, 'all_but_last': freeze_all_but_last, 'middle': freeze_middle}

    if cfg.net == 'DenseNet121':
        net = DenseNet121
    elif cfg.net == 'ResNet50':
        net = ResNet50

    # todo define client ids

    # federated learning parameters
    num_clients = len(chexpert_client_ids) + len(mendeley_client_ids)
    if cfg.training.max_client_selections is not None:
        assert (num_clients * cfg.training.max_client_selections >=
                round(num_clients * cfg.training.q) * cfg.training.T), "Client fraction or maximum rounds for client selection is not large enough."

    if cfg.dp.enabled:
        assert cfg.model.freeze_layers == 'batch_norm' or cfg.model.freeze_layers == 'all_but_last', "Batch norm layers must be frozen for private training."

    # initialize client instances and their datasets
    clients = init_clients(cfg.data, cfg.training.B, cfg.model.pre_trained,
                           chexpert_client_ids, mendeley_client_ids)

    # show images for testing
    # for batch in clients[0].train_loader:
    #     transforms.ToPILImage()(batch[0][0]).show()
    #     print(batch[1][0])
    #
    #  for batch in clients[1].val_loader:
    #     transforms.ToPILImage()(batch[0][0]).show()
     #     print(batch[1][0])

     # get labels and indices of data from dataloaders
     # modify chexpert_data dataloader to also return indices for this
     # for i in [12,13,14,15,16,17,18,19]:
     #     print("Client ", i)
     #     for batch in clients[i].train_loader:
     #         print("Labels: ", batch[1])
     #         print("Inidces: ", batch[2])


    # create global model
    if use_gpu:
        global_model = net(nnClassCount, cfg.data.colour_input, cfg.model.pre_trained).cuda()
        # model=torch.nn.DataParallel(model).cuda()
    else:
        global_model = net(nnClassCount, cfg.data.colour_input, cfg.model.pre_trained)

    if cfg.model.checkpoint is not None: # load weights if some model is specified
        checkpoint = torch.load(cfg.model.checkpoint)
        if 'state_dict' in checkpoint:
            global_model.load_state_dict(checkpoint['state_dict'])
        else:
            global_model.load_state_dict(checkpoint)

    # freeze batch norm layers already, so it passes the privacy engine checks
    if cfg.model.freeze_layers != 'none':
         freeze_dict[cfg.model.freeze_layers](global_model)

    # define path to store results in
    output_path = check_path(cfg.output_path, warn_exists=True)

    # save initial global model parameters
    torch.save({'state_dict': global_model.state_dict()}, output_path + 'global_init.pth.tar')

    # initialize client models and optimizers
    for client_k in clients:
        print(f"Initializing model and optimizer of {client_k.name}")
        client_k.model = copy.deepcopy(global_model)
        client_k.init_optimizer(cfg)

        if cfg.dp.enabled:
            # compute personal delta dependent on client's dataset size, or choose min delta value allowed
            client_k.delta = min(cfg.dp.min_delta, 1 / client_k.n_data * 0.1)
            print(f'Client delta: {client_k.delta}')
            # attach DP privacy engine for private training
            client_k.privacy_engine = opacus.PrivacyEngine(client_k.model,
                                                           target_epsilon=cfg.dp.epsilon,
                                                           target_delta=client_k.delta,
                                                           max_grad_norm=cfg.dp.S,
                                                           epochs=cfg.training.max_client_selections,
                                                           # noise_multiplier=cfg.dp.z,
                                                           # get sample rate with respect to client's dataset
                                                           sample_rate=min(1, cfg.training.batch_size / client_k.n_data))
            client_k.privacy_engine.attach(client_k.optimizer)

    # =========== DO FEDERATED LEARNING =============
    fed_start = time.time()
    global_auc = []
    best_global_auc = 0
    track_no_improv = 0
    client_pool = clients # clients that may be selected for training

    for i in range(cfg.training.T):

        print(f"[[[ Round {i} Start ]]]")

        # Step 1: select random fraction of clients
        sel_clients = random.sample(client_pool, round(num_clients * cfg.training.q))
        for sel_client in sel_clients:
            sel_client.selected_rounds += 1
        # check if clients have now exceeded the maximum number of rounds they can be selected
        # and drop them from the pool if so
        for cp in client_pool:
            if cp.selected_rounds == cfg.training.max_client_selections:
                client_pool.remove(cp)
        print("Number of selected clients: ", len(sel_clients))
        print(f"Clients selected: {[sel_cl.name for sel_cl in sel_clients]}")

        # Step 2: send global model to clients and train locally
        for client_k in sel_clients:
            # reset model at client's site
            client_k.model_params = None
            # https://blog.openmined.org/pysyft-opacus-federated-learning-with-differential-privacy/
            with torch.no_grad():
                for client_params, global_params in zip(client_k.model.parameters(), global_model.parameters()):
                    client_params.set_(copy.deepcopy(global_params))

            print(f"<< {client_k.name} Training Start >>")
            # set output path for storing models and results
            client_k.output_path = output_path + f"round{i}_{client_k.name}/"
            client_k.output_path = check_path(client_k.output_path, warn_exists=False)
            print(client_k.output_path)

            # Step 3: Perform local computations
            # returns local best model
            train_valid_start = time.time()
            Trainer.train(client_k, cfg, use_gpu, out_csv=f"round{i}_{client_k.name}.csv",
                          freeze_mode=cfg.model.freeze_layers)
            client_k.model_params = client_k.model.state_dict().copy()
            client_time = round(time.time() - train_valid_start)
            print(f"<< {client_k.name} Training Time: {client_time} seconds >>")

        first_cl = sel_clients[0]
        # Step 4: return updates to server
        for key in first_cl.model_params:
            # iterate through parameters layer-wise
            weights, weight_n = [], []

            for cl in sel_clients:
                if cfg.training.weighted_avg:
                    weights.append(cl.model_params[key] * cl.n_data)
                    weight_n.append(cl.n_data)
                else:
                    weights.append(cl.model_params[key])
                    weight_n.append(1)
            # store parameters with first client for convenience
            first_cl.model_params[key] = sum(weights) / sum(weight_n) # weighted averaging model weights

        # Step 5: server updates global state
        global_model.load_state_dict(first_cl.model_params)

        # also save intermediate models
        torch.save(global_model.state_dict(),
                   output_path + f"global_{i}rounds.pth.tar")

        # validate global model on client validation data
        print("Validating global model...")
        global_auroc = []
        for cl in clients:
            if cl.val_loader is not None:
                ground_truth, predictions, client_auroc, _ = Trainer.test(global_model, cl.val_loader,
                                                                          cfg.data.class_idx, use_gpu, checkpoint=None)
                global_auroc.append(client_auroc)
            else:
                global_auroc.append(np.nan)
           #  print(GT)
           # print(PRED)
        cur_global_auc = np.nanmean(np.array(global_auroc))
        print("AUC Mean: {:.3f}".format(cur_global_auc))
        global_auc.append(cur_global_auc)

        # track early stopping & lr decay
        if cur_global_auc > best_global_auc:
            best_global_auc = cur_global_auc
            track_no_improv = 0
        else:
            track_no_improv += 1
            if track_no_improv == reduce_lr_rounds:
                # decay lr
                cfg.lr = cfg.lr * 0.1
                print(f"Learning rate reduced to {cfg.lr}")
            elif track_no_improv == earl_stop_rounds:
                print(f'Global AUC has not improved for {earl_stop_rounds} rounds. Stopping training.')
                break

        print(f"[[[ Round {i} End ]]]\n")

    # save global AUC in CSV
    all_metrics = [list(range(com_rounds)), global_auc]
    with open(output_path+'global_validation.csv', 'w') as f:
        header = ['round', 'val AUC']
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(zip(*all_metrics))

    print("Global model trained")
    fed_end = time.time()
    print(f"Total training time: {round(fed_end - fed_start)}")


    # merge local metrics to CSV
    try:
        merge_eval_csv(output_path, out_file='train_results.csv')
    except:
        print("Merging result CSVs failed. Try again manually.")


def init_clients(data_cfg, batch_size, pre_trained, chexpert_client_ids, mendeley_client_ids):
    train_transform_sequence, test_transform_sequence = get_dataset_transforms(data_cfg, pre_trained)
    clients = ([Client(name=f'CXP_client_{n}') for n in chexpert_client_ids] +
               [Client(name=f'MDL_client_{n}') for n in mendeley_client_ids])
    num_clients = len(chexpert_client_ids) + len(mendeley_client_ids)

    for i in range(num_clients):
        cur_client = clients[i]
        print(f"Initializing {cur_client.name}")

        if 'CXP' in cur_client.name:
            data_path = CHEXPERT_PATH
            path_to_client = check_path(CHEXPERT_CLIENT_DATA + cur_client.name.replace('CXP_', ''),
                                    warn_exists=False, require_exists=True)
            train_file = path_to_client + '/client_train.csv'
        else:
            data_path = MENDELEY_PATH
            path_to_client = check_path(MENDELEY_CLIENT_DATA + cur_client.name.replace('MDL_', ''),
                                    warn_exists=False, require_exists=True)
            train_file = path_to_client + '/client_train.csv'

        cur_client.train_data = CheXpertDataSet(data_path, train_file, data_cfg.class_idx, data_cfg.policy,
                                                colour_input=data_cfg.colour_input, transform=train_transform_sequence)

        assert cur_client.train_data[0][0].shape == torch.Size([len(data_cfg.colour_input), data_cfg.resize, data_cfg.resize])
        assert cur_client.train_data[0][1].shape == torch.Size([len(data_cfg.class_idx)])

        cur_client.n_data = len(cur_client.train_data)
        print(f"Holds {cur_client.n_data} data points")

        # drop last incomplete batch if dataset has at least one full batch, otherwise keep one incomplete batch
        if  cur_client.n_data > batch_size:
            drop_last = True
            print(f"Dropping incomplete batch of {cur_client.n_data % batch_size} data points")
            cur_client.n_data = cur_client.n_data - cur_client.n_data % batch_size
        else:
            drop_last = False

        cur_client.train_loader = DataLoader(dataset=cur_client.train_data, batch_size=batch_size, shuffle=True,
                                            num_workers=4, pin_memory=True, drop_last=drop_last)

        val_file = path_to_client + 'client_val.csv'
        test_file = path_to_client + 'client_test.csv'

        if os.path.exists(val_file):
            cur_client.val_data = CheXpertDataSet(data_path, val_file, data_cfg.class_idx, data_cfg.policy,
                                                  colour_input=data_cfg.colour_input, transform=test_transform_sequence)
            cur_client.test_data = CheXpertDataSet(data_path, test_file, data_cfg.class_idx, data_cfg.policy,
                                                   colour_input=data_cfg.colour_input, transform=test_transform_sequence)

            cur_client.val_loader = DataLoader(dataset=cur_client.val_data, batch_size=batch_size, shuffle=False,
                                                num_workers=4, pin_memory=True)
            cur_client.test_loader = DataLoader(dataset = cur_client.test_data, num_workers = 4, pin_memory = True)

        else: # clients that don't
            print(f"No validation data for client{i}")
            cur_client.val_loader = None
            cur_client.test_loader = None

    return clients


def get_dataset_transforms(dataset_cfg: DictConfig, pre_trained: bool):
    #define mean and std dependent on whether using a pretrained model
    if pre_trained:
        data_mean = IMAGENET_MEAN
        data_std = IMAGENET_STD
    else:
        data_mean = CHEXPERT_MEAN
        data_std = CHEXPERT_STD

    if dataset_cfg.colour_input == 'L':
        data_mean = np.mean(data_mean)
        data_std = np.mean(data_std)

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
    print(f"{torch.cuda.device_count()} GPUs available")

    return True




if __name__ == "__main__":
    main()
