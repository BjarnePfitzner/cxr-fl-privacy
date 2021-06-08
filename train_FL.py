
"""Train a model using federated learning"""

#set which GPUs to use
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3' #configure this

import pandas as pd
import argparse
import json
from PIL import Image
import time

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
use_gpu = torch.cuda.is_available()

#local imports
from chexpert_data import CheXpertDataSet
from trainer import Trainer
from trainer import DenseNet121


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)


def main():

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #parse config file
    parser.add_argument('cfg_path', type = str, help = 'Path to the config file in json format.')
    #output path for storing results
    parser.add_argument('--output_path', '-o', help = 'Path to save results.', default = 'results/')
    #whether to assert GPU usage (disable for testing without GPU)
    parser.add_argument('--no_gpu', dest='no_gpu', help='Don\'t verify GPU usage.', action='store_true')
    args = parser.parse_args()
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if not args.no_gpu:
        check_gpu_usage(use_gpu)

    #TODO configure randomness
    random_seed = cfg['random_seed']
    torch.manual_seed(random_seed)

    # Parameters from config file
    nnIsTrained = cfg['pre_trained']     # pre-trained using ImageNet
    trBatchSize = cfg['batch_size']
    trMaxEpoch = cfg['max_epochs']

    # Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = cfg['imgtransResize']
    # imgtransCrop = cfg['imgtransCrop']
    policy = cfg['policy']

    nnClassCount = cfg['nnClassCount']       # dimension of the output
    class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    #run preprocessing to obtain these files
    pathFileTrain = './CheXpert-v1.0-small/train_mod.csv'
    pathFileValid = './CheXpert-v1.0-small/valid_mod.csv'
    pathFileTest = './CheXpert-v1.0-small/test_mod.csv'

    # define transforms
    # if using augmentation, use different transforms for training, test & val data
    transformSequence = transforms.Compose([transforms.Resize((imgtransResize,imgtransResize)),
                                            # transforms.RandomResizedCrop(imgtransCrop),
                                            # transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                                            ])

    # Load dataset
    datasetTrain = CheXpertDataSet(pathFileTrain, nnClassCount, policy, transform = transformSequence)
    print("Train data length:", len(datasetTrain))

    #remove transformations here?
    datasetValid = CheXpertDataSet(pathFileValid, nnClassCount, policy, transform = transformSequence)
    print("Valid data length:", len(datasetValid))

    datasetTest = CheXpertDataSet(pathFileTest, nnClassCount, policy, transform = transformSequence)
    print("Test data length:", len(datasetTest))

    assert datasetTrain[0][0].shape == torch.Size([3,imgtransResize,imgtransResize])
    assert datasetTrain[0][1].shape == torch.Size([nnClassCount])

    #Create dataLoaders
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True, num_workers=2, pin_memory=True)
    dataLoaderVal = DataLoader(dataset = datasetValid, batch_size = trBatchSize, num_workers = 2, pin_memory = True)
    dataLoaderTest = DataLoader(dataset = datasetTest, num_workers = 2, pin_memory = True)

    print('Length train dataloader (n batches): ', len(dataLoaderTrain))

    model = DenseNet121(nnClassCount, cfg['pre_trained'])

    #train the model
    output_path = args.output_path
    if args.output_path[-1] != '/':
        output_path = args.output_path + '/'
    else:
        output_path= args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    # start = time.time()
    # model_num, params = Trainer.train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, cfg, output_path, use_gpu)
    # end = time.time()
    # print(end-start)

    # outGT, outPRED = Trainer.test(model, dataLoaderTest, nnClassCount, class_names, use_gpu,
    #                                     checkpoint='results/1-epoch_FL.pth.tar')




def check_gpu_usage(use_gpu):
    assert use_gpu is True, "GPU not used"
    assert torch.cuda.device_count() == len(os.environ["CUDA_VISIBLE_DEVICES"]), "Wrong number of GPUs available to Pytorch"
    print(f"{torch.cuda.device_count} GPUs available")

    return True


if __name__ == "__main__":
    main()
