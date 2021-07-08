
"""Validate a model on each client's validation or test data."""

#set which GPUs to use
import os
selected_gpus = [7] #configure this
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in selected_gpus])

import pandas as pd
import argparse
import json
from PIL import Image
import time
import random
import numpy as np
import csv

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#local imports
from chexpert_data import CheXpertDataSet
from trainer import Trainer, DenseNet121, Client
from utils import check_path

CSV_OUTPUT_NAME = 'test_model.csv' # name for file in which to store results

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)

CHEXPERT_MEAN = [0.5029, 0.5029, 0.5029]
CHEXPERT_STD = [0.2899, 0.2899, 0.2899]


def main():
    use_gpu = torch.cuda.is_available()

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #parse config file
    parser.add_argument('cfg_path', type = str, help = 'Path to the config file in json format.')
    #model checkpoint path
    parser.add_argument('--model', '-m', dest='model_path', help='Path to model.', required=True)
    #output path for storing results
    parser.add_argument('--output_path', '-o', help = 'Path to save results.', default = 'results/')
    #set path to chexpert data
    parser.add_argument('--chexpert', '-d', dest='chexpert_path', help='Path to CheXpert data.', default='./')
    #whether to assert GPU usage (disable for testing without GPU)
    parser.add_argument('--no_gpu', dest='no_gpu', help='Don\'t verify GPU usage.', action='store_true')
    parser.add_argument('--val', dest='use_val', help='Whether to use validation data. Test data is used by default.', action='store_true')


    args = parser.parse_args()
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if not args.no_gpu:
        check_gpu_usage(use_gpu)
    else:
        use_gpu=False

    #only use pytorch randomness for direct usage with pytorch
    #check for pitfalls when using other modules
    random_seed = cfg['random_seed']
    if random_seed != None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(random_seed)


    # Parameters from config file, client training
    nnIsTrained = cfg['pre_trained']     # pre-trained using ImageNet
    trBatchSize = cfg['batch_size']
    trMaxEpoch = cfg['max_epochs']

    # Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = cfg['imgtransResize']
    # imgtransCrop = cfg['imgtransCrop']
    policy = cfg['policy']

    class_idx = cfg['class_idx'] #indices of classes used for classification
    nnClassCount = len(class_idx)       # dimension of the output


    #federated learning parameters
    num_clients = cfg['num_clients']
    client_dirs = cfg['client_dirs']
    assert num_clients == len(client_dirs), "Number of clients doesn't correspond to number of directories specified"
    fraction = cfg['fraction']
    com_rounds = cfg['com_rounds']

    data_path = check_path(args.chexpert_path, warn_exists=False, require_exists=True)

    #define mean and std dependent on whether using a pretrained model
    if nnIsTrained:
        data_mean = IMAGENET_MEAN
        data_std = IMAGENET_STD
    else:
        data_mean = CHEXPERT_MEAN
        data_std = CHEXPERT_STD

    # define transforms
    # if using augmentation, use different transforms for training, test & val data
    train_transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                            # transforms.RandomResizedCrop(imgtransResize),
                                            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(data_mean, data_std)
                                            ])
    test_transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(data_mean, data_std)
                                            ])


    #initialize client instances
    clients = [Client(name=f'client{n}') for n in range(num_clients)]
    for i in range(num_clients):
        cur_client = clients[i]
        print(f"Initializing {cur_client.name}")

        path_to_client = check_path(data_path + 'CheXpert-v1.0-small/' + client_dirs[i], warn_exists=False, require_exists=True)

        cur_client.train_file = path_to_client + 'client_train.csv'
        cur_client.val_file = path_to_client + 'client_val.csv'
        cur_client.test_file = path_to_client + 'client_test.csv'

        cur_client.train_data = CheXpertDataSet(data_path, cur_client.train_file, class_idx, policy, transform = train_transformSequence)
        cur_client.val_data = CheXpertDataSet(data_path, cur_client.val_file, class_idx, policy, transform = test_transformSequence)
        cur_client.test_data = CheXpertDataSet(data_path, cur_client.test_file, class_idx, policy, transform = test_transformSequence)

        assert cur_client.train_data[0][0].shape == torch.Size([3,imgtransResize,imgtransResize])
        assert cur_client.train_data[0][1].shape == torch.Size([nnClassCount])

        cur_client.n_data = cur_client.get_data_len()
        print(f"Holds {cur_client.n_data} data points")

        cur_client.train_loader = DataLoader(dataset=cur_client.train_data, batch_size=trBatchSize, shuffle=True,
                                            num_workers=4, pin_memory=True)
        # assert cur_client.train_loader.dataset == cur_client.train_data

        cur_client.val_loader = DataLoader(dataset=cur_client.val_data, batch_size=trBatchSize, shuffle=True,
                                            num_workers=4, pin_memory=True)
        cur_client.test_loader = DataLoader(dataset = cur_client.test_data, num_workers = 4, pin_memory = True)

    # show images for testing
    # for batch in clients[0].train_loader:
    #     transforms.ToPILImage()(batch[0][0]).show()
    #     print(batch[1][0])
    #
    # for batch in clients[0].val_loader:
    #     transforms.ToPILImage()(batch[0][0]).show()
    #     print(batch[1][0])


    # create model
    if use_gpu:
        model = DenseNet121(nnClassCount, pre_trained=False).cuda()
        model=torch.nn.DataParallel(model).cuda()
    else:
        model = DenseNet121(nnClassCount, pre_trained=False)

    # define path to store results in
    output_path = check_path(args.output_path, warn_exists=True)

    # read model checkpoint
    checkpoint = args.model_path
    modelCheckpoint = torch.load(checkpoint)
    if 'state_dict' in modelCheckpoint:
        model.load_state_dict(modelCheckpoint['state_dict'])
    else:
        model.load_state_dict(modelCheckpoint)

    #validate global model on client validation data
    print("Validating model on each client's data...")

    aurocMean_global_clients = [] # list of AUCs of clients

    # check if validation or test data should be used
    if args.use_val:
        print('Using validation data')
        for cl in clients:
            _, _, cl_aurocMean = Trainer.test(model, cl.val_loader, class_idx, use_gpu, checkpoint=None)
            aurocMean_global_clients.append(cl_aurocMean)
    else:
        for cl in clients:
            _, _, cl_aurocMean = Trainer.test(model, cl.test_loader, class_idx, use_gpu, checkpoint=None)
            aurocMean_global_clients.append(cl_aurocMean)

    # mean of client AUCs
    auc_global = np.array(aurocMean_global_clients).mean()
    print("AUC Mean of all clients: {:.3f}".format(auc_global))
    aurocMean_global_clients.append(auc_global) # save mean
    save_clients = [cl.name for cl in clients]
    save_clients.append('avg')

    # save AUC in CSV
    print(f'Saving in {output_path+CSV_OUTPUT_NAME}') 
    all_metrics = [save_clients, aurocMean_global_clients]
    with open(output_path+CSV_OUTPUT_NAME, 'w') as f:
        header = ['client', 'AUC']
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(zip(*all_metrics))


def check_gpu_usage(use_gpu):

    """Give feedback to whether GPU is available and if the expected number of GPUs are visible to PyTorch.
    """
    assert use_gpu is True, "GPU not used"
    assert torch.cuda.device_count() == len(selected_gpus), "Wrong number of GPUs available to Pytorch"
    print(f"{torch.cuda.device_count()} GPUs available")

    return True




if __name__ == "__main__":
    main()