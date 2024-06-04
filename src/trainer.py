"""Module for training.
Originally based on and modified from https://github.com/Stomper10/CheXpert/blob/master/materials.py.

Contains:
    A Trainer class which bundles relevant functions for training, validation, testing.
    Custom modules for DenseNet121 and ResNet50.
    Functions for layer freezing.
    A Client class for managing individual FL clients.
"""

import time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, auc, precision_recall_curve, accuracy_score, matthews_corrcoef
from tqdm import tqdm
import csv
import logging
import warnings

import torch

from src.utils.model_utils import freeze_all_but_last, freeze_batchnorm, freeze_middle


class Trainer:

    @staticmethod
    def train(client_k, cfg, use_gpu, out_csv='train_log.csv', checkpoint=None, freeze_mode='none'):

        """Train a local model from a client instance.
        Args:
            client_k (Client object): Client instance with client model, data laoders, output path attributes.
            cfg (DictConfig): Config dictionary containing training parameters.
            use_gpu (bool): Whether to use available GPUs.
            out_csv (str): Name of CSV file used for logging. Stored in output path.
            checkpoint (str): A model checkpoint to load from for continuing training.
            freeze_mode (str): Information about which layers to freeze during training.
        Returns nothing.
        """
        if cfg.save_model:
            out_csv_path = client_k.output_path + out_csv

        loss = torch.nn.BCELoss() # setting binary cross entropy as loss function

        if checkpoint is not None: # load checkpoint
            model_checkpoint = torch.load(checkpoint)
            client_k.model.load_state_dict(model_checkpoint['state_dict'])
            client_k.optimizer.load_state_dict(model_checkpoint['optimizer'])
        else:
            # reset optimizer state (for Adam) and potentially reduce LR
            client_k.init_optimizer(cfg.training)

        # logging metrics
        loss_min = 100000
        train_start = []
        train_end = []

        save_epoch = []
        save_train_loss = []
        save_val_loss = []
        save_val_metrics = []
        metrics_names = ['auroc', 'auprc', 'auprc_macroavg', 'f1', 'accuracy', 'mcc']
        save_epsilon = []
        save_alpha = []
        save_delta = []

        # train model for number of epochs
        for epochID in range(0, cfg.training.E):
            logging.debug(f'Start epoch {epochID}')
            train_start.append(time.time())
            losst = Trainer.epoch_train(client_k.model, client_k.train_loader, client_k.optimizer, loss, use_gpu, freeze_mode=freeze_mode)
            train_end.append(time.time())

            # model validation
            if client_k.val_loader is not None:
                logging.debug("Validating model...")
                lossv, metrics_mean = Trainer.epoch_val(client_k.model, client_k.val_loader, loss, use_gpu)
                logging.debug(f"Training loss: {losst:.3f}, Valid loss: {lossv:.3f}")
            else:
                # if the client doesn't have validation data, add nan placeholders to metrics
                lossv, metrics_mean = (np.nan, {key: np.nan for key in metrics_names})

            # save model to intermediate checkpoint file
            model_num = epochID + 1
            if cfg.save_model:
                # todo hacky way of only saving model in epochs 0, 4 and 9
                if any([f'round{the_round}_' in out_csv for the_round in [0, 4, 9]]):
                    torch.save({'epoch': model_num, 'state_dict': client_k.model.state_dict(),
                                'loss': loss_min, 'optimizer' : client_k.optimizer.state_dict()},
                               f"{client_k.output_path}{model_num}-epoch_FL.pth.tar")

            # keep parameters of best model
            if lossv < loss_min:
                loss_min = lossv
                logging.debug(f'Epoch {str(model_num)} [++++] val loss decreased')
            else:
                logging.debug(f'Epoch {str(model_num)} [----] val loss did not decrease or no val data available')

            # track metrics
            save_epoch.append(model_num)
            save_train_loss.append(losst)
            save_val_loss.append(lossv)
            save_val_metrics.append(metrics_mean)

            if cfg.training.track_norm:
                # follow L2 grad norm per parameter layer
                grad_norm = []
                for p in list(filter(lambda p: p.grad is not None, client_k.model.parameters())):
                    cur_norm = p.grad.data.norm(2).item()
                    grad_norm.append(cur_norm)

            if cfg.dp.enabled:
                epsilon, best_alpha = client_k.privacy_engine.get_privacy_spent()
                logging.debug(f"epsilon: {epsilon:.2f}, best alpha: {best_alpha}")
                save_epsilon.append(epsilon)
                save_alpha.append(best_alpha)
                save_delta.append(client_k.delta)

        train_time = np.array(train_end) - np.array(train_start)
        logging.debug(f"Training time for each epoch: {train_time.round(0)} seconds")

        # save logging metrics in CSV
        reordered_metrics = [[metric_dict[key] for metric_dict in save_val_metrics] for key in metrics_names]
        all_metrics = [save_epoch, train_time, save_train_loss, save_val_loss] + reordered_metrics
        if cfg.training.track_norm:
            all_metrics += [[grad_norm]]
        if cfg.dp.enabled:
            all_metrics += [save_epsilon, save_delta, save_alpha]

        if cfg.save_model:
            with open(out_csv_path, 'w') as f:
                header = ['epoch', 'time', 'train loss', 'val loss'] + metrics_names
                if cfg.training.track_norm:
                    header += ['track_norm']
                if cfg.dp.enabled:
                    header += ['epsilon', 'best_alpha', 'delta']
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(zip(*all_metrics))

        return all_metrics

    @staticmethod
    def epoch_train(model, data_loader_train, optimizer, loss, use_gpu, freeze_mode='none'):

        """Train a model for one epoch.
        Args:
            model (model object): Model to train.
            data_loader_train (dataloader object): PyTorch dataloader with training data.
            optimizer (optimizer object): Optimizer instance.
            loss (function object): Loss function.
            use_gpu (bool): Whether to train on GPU.
            freeze_mode (str): Information about which layers to freeze.
        Returns:
            (float) Mean training loss over batches."""

        losstrain = 0
        model.train()

        if freeze_mode == 'batch_norm':
            freeze_batchnorm(model)
        if freeze_mode == 'all_but_last':
            freeze_all_but_last(model)
        if freeze_mode == 'middle':
            freeze_middle(model)

        # usual training procedure
        #with tqdm(data_loader_train, unit='batch') as tqdm_loader:

            #for varInput, target in tqdm_loader:
        for varInput, target in data_loader_train:

            if use_gpu:
                target = target.cuda(non_blocking=True)
                varInput = varInput.cuda(non_blocking=True)

            varOutput = model(varInput) #forward pass
            lossvalue = loss(varOutput, target)

            optimizer.zero_grad() #reset gradient
            lossvalue.backward()
            optimizer.step()

            losstrain += lossvalue.item()

                #tqdm_loader.set_postfix(loss=lossvalue.item())

        return losstrain / len(data_loader_train)


    @staticmethod
    def epoch_val(model, data_loader_val, loss, use_gpu):

        """Validate a model.
        Args:
            model (model object): Model to validate.
            data_loader_val (dataloader object): PyTorch ataloader with validation data.
            loss (function object): Loss function.
            use_gpu (bool): Whether to train on GPU.
        Returns:
            (float): Mean validation loss over batches.
            (float): Mean AUROC over all labels."""

        model.eval()
        lossVal = 0

        if use_gpu:
            out_gt = torch.FloatTensor().cuda()
            out_pred = torch.FloatTensor().cuda()
        else:
            out_gt = torch.FloatTensor()
            out_pred = torch.FloatTensor()

        with torch.no_grad():
            for model_input, target in data_loader_val:

                if use_gpu:
                    target = target.cuda(non_blocking=True)
                    model_input = model_input.cuda(non_blocking=True)

                model_output = model(model_input)

                lossVal += loss(model_output, target).item()

                # collect predictions and ground truth for AUROC computation
                out_gt = torch.cat((out_gt, target), 0)
                out_pred = torch.cat((out_pred, model_output), 0)

        # compute metrics
        metrics_per_class = Trainer.compute_metrics(out_gt, out_pred)
        metrics_mean = {key: np.nanmean(np.array(val)) for key, val in metrics_per_class.items()}

        for metric_name, metric_val in metrics_mean.items():
            logging.debug(f'{metric_name} mean: {metric_val:.3f}')

        return lossVal / len(data_loader_val), metrics_mean


    @staticmethod
    def test(model, data_loader_test, use_gpu, checkpoint=None):

        """Stand-alone function for testing a model.
        Args:
            model (model object): Model to validate.
            data_loader_test (dataloader object): PyTorch ataloader with test data.
            class_idx (list): List of label indices with which the model has been trained.
            use_gpu (bool): Whether to train on GPU.
            checkpoint (str): A model checkpoint to load parameters from.
        Returns:
            (tensor): Ground truth labels.
            (tensor): Predicted labels.
            (float) Mean AUROC over all labels.
            (list): Individual AUROC for each label."""

        model.eval()

        if use_gpu:
            out_gt = torch.FloatTensor().cuda()
            out_pred = torch.FloatTensor().cuda()
        else:
            out_gt = torch.FloatTensor()
            out_pred = torch.FloatTensor()

        if checkpoint is not None:
            model_checkpoint = torch.load(checkpoint)
            if 'state_dict' in model_checkpoint:
                model.load_state_dict(model_checkpoint['state_dict'])
            else:
                model.load_state_dict(model_checkpoint)

        with torch.no_grad():
            for model_input, target in data_loader_test:

                if use_gpu:
                    target = target.cuda(non_blocking=True)
                    model_input = model_input.cuda(non_blocking=True)
                out_gt = torch.cat((out_gt, target), 0)

                out = model(model_input)
                out_pred = torch.cat((out_pred, out), 0)

        # compute metrics
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            metrics_per_class = Trainer.compute_metrics(out_gt, out_pred)
            metrics_mean = {key: np.nanmean(np.array(val)) for key, val in metrics_per_class.items()}

        return out_gt, out_pred, metrics_mean, metrics_per_class


    @staticmethod
    def compute_metrics(data_gt, data_pred):
        """Compute Metrics.
        Args:
            data_gt (tensor): Ground truth labels.
            data_pred (tensor): Predicted labels.
        Returns:
            (dict): metric: list of metric values for each label."""

        def compute_mccs_accs(y_true, y_pred_probs, num_thresholds):
            # Compute `num_thresholds` thresholds in [0, 1]
            if num_thresholds == 1:
                thresholds = [0.5]
            else:
                thresholds = [
                    (i + 1) * 1.0 / (num_thresholds - 1)
                    for i in range(num_thresholds - 2)
                ]
                thresholds = [0.0] + thresholds + [1.0]

            mcc_scores = []
            acc_scores = []
            for threshold in thresholds:
                y_pred = (y_pred_probs >= threshold).astype(int)
                mcc_scores.append(matthews_corrcoef(y_true, y_pred))
                acc_scores.append(accuracy_score(y_true, y_pred))
            return mcc_scores, acc_scores

        metrics_dict = {
            'auroc': [],
            'auprc': [],
            'auprc_macroavg': [],
            'f1': [],
            #'f1_macroavg': [],
            'accuracy': [],
            'mcc': [],
            #'threshold': []
        }
        data_np_gt = data_gt.cpu().numpy()
        data_np_pred = data_pred.cpu().numpy()
        nn_class_count = data_gt.shape[1] # [0] is the batch size   # todo does not work with two output neurons

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            for i in range(nn_class_count):
                try:
                    metrics_dict['auroc'].append(roc_auc_score(data_np_gt[:, i], data_np_pred[:, i]))
                except ValueError:
                    logging.debug(f"AUROC not defined for label {i}")
                    metrics_dict['auroc'].append(np.nan)

                #threshold = 0.5
                try:
                    prc, rec, thresholds = precision_recall_curve(data_np_gt[:, i], data_np_pred[:, i])
                    metrics_dict['auprc'].append(auc(rec, prc))

                    prc2, rec2, _ = precision_recall_curve(data_np_gt[:, i], data_np_pred[:, i], pos_label=0)
                    metrics_dict['auprc_macroavg'].append(np.mean([auc(rec2, prc2), auc(rec, prc)]))

                    f1_scores = 2 * rec * prc / (rec + prc)
                    #threshold = thresholds[np.argmax(f1_scores)]
                    metrics_dict['f1'].append(np.max(f1_scores))

                    # todo does not work, since length don't necessarily match
                    #f1_scores2 = 2 * rec2 * prc2 / (rec2 + prc2)
                    #macro_f1_scores = (f1_scores + f1_scores2) / 2
                    #metrics_dict['f1_macroavg'].append(np.max(macro_f1_scores))
                except ValueError:
                    logging.debug(f"AUPRC and/or F1 not defined for label {i}")
                    metrics_dict['auprc'].append(np.nan)
                    metrics_dict['auprc_macroavg'].append(np.nan)
                    metrics_dict['f1'].append(np.nan)
                    #metrics_dict['f1_macroavg'].append(np.nan)

                #metrics_dict['threshold'].append(threshold)
                #data_np_classes = [1 if i > threshold else 0 for i in data_np_pred[:, i]]
                #metrics_dict['accuracy'] = accuracy_score(data_np_gt[:, i], data_np_classes)
                try:
                    mccs, accs = compute_mccs_accs(data_np_gt[:, i], data_np_pred[:, i], num_thresholds=200)
                    metrics_dict['mcc'].append(np.max(mccs))
                    metrics_dict['accuracy'].append(np.max(accs))
                except ValueError:
                    logging.debug(f"MCC not defined for label {i}")
                    metrics_dict['mcc'].append(np.nan)
                    metrics_dict['accuracy'].append(np.nan)

        return metrics_dict
