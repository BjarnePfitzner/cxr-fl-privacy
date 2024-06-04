import logging
import wandb
import numpy as np
import torch

from src.trainer import Trainer

def evaluate_model(model, clients, epoch, val_test, use_gpu, cfg, plot_curves=False):
    all_client_metrics = {}
    all_client_labels = {}
    all_client_preds = {}
    # num_no_val = 0
    for cl in clients:
        if val_test == 'val':
            data_loader = cl.val_loader
        else:
            data_loader = cl.test_loader
        if data_loader is not None:
            # get AUC mean and per class for current client
            labels, preds, client_metrics, client_metrics_per_class = Trainer.test(model, data_loader, use_gpu)
            all_client_metrics[cl.name] = client_metrics
            all_client_labels[cl.name] = labels.cpu().numpy()
            all_client_preds[cl.name] = preds.cpu().numpy()
            # for i in range(len(cfg.data.class_idx)):
            #    # track sum per class over all clients
            #    aurocMean_individual_clients[i] += client_metrics_per_class[i]
        else:
            all_client_metrics[cl.name] = {key: np.nan for key in
                                                ['auroc', 'auprc', 'auprc_macroavg', 'f1', 'accuracy', 'mcc']}
            logging.debug(f"No test data available for {cl.name}")
            # num_no_val += 1

    es_metric_val = log_metrics_to_wandb(all_client_metrics, epoch, test_set_key=val_test,
                                         key_metric=cfg.training.early_stopping_metric,
                                         plot_per_client_metrics=cfg.evaluation.log_local_performance,
                                         plot_histogram=cfg.evaluation.log_local_histograms)
    logging.info(f"Macro {cfg.training.early_stopping_metric} mean of all clients: {es_metric_val:.3f}")
    # aurocMean_global_clients.append(auc_global)  # save global mean

    all_labels = np.concatenate(list(all_client_labels.values()))
    all_preds = np.concatenate(list(all_client_preds.values()))
    global_micro_metrics = Trainer.compute_metrics(torch.FloatTensor(all_labels), torch.FloatTensor(all_preds))
    global_micro_metrics = {key: np.nanmean(np.array(val)) for key, val in global_micro_metrics.items()}
    wandb.log({
        f'global/micro/{val_test}_{metric_name}': metric_val
        for metric_name, metric_val in global_micro_metrics.items()
    }, step=epoch)

    for base_dataset_name in ['CXP', 'MDL']:
        if any([base_dataset_name in client_name for client_name in all_client_metrics.keys()]):
            dataset_labels = np.concatenate([client_labels for client_name, client_labels in all_client_labels.items()
                                             if base_dataset_name in client_name])
            dataset_preds = np.concatenate([client_preds for client_name, client_preds in all_client_preds.items()
                                            if base_dataset_name in client_name])
            global_micro_metrics = Trainer.compute_metrics(torch.FloatTensor(dataset_labels),
                                                           torch.FloatTensor(dataset_preds))
            global_micro_metrics = {key: np.nanmean(np.array(val)) for key, val in global_micro_metrics.items()}
            wandb.log({
                f'{base_dataset_name}/micro/{val_test}_{metric_name}': metric_val
                for metric_name, metric_val in global_micro_metrics.items()
            }, step=epoch)

    # Plot overall ROC and PRC
    if plot_curves:
        expanded_preds = np.concatenate((1 - np.array(all_preds), np.array(all_preds)), axis=-1)
        wandb.log({
            f'global/{val_test}_roc': wandb.plot.roc_curve(y_true=all_labels, y_probas=expanded_preds,
                                                           classes_to_plot=[0, 1], labels=['No Finding', 'Finding'],
                                                           split_table=True),
        }, step=epoch)
        wandb.log({
            f'global/{val_test}_prc': wandb.plot.pr_curve(y_true=all_labels, y_probas=expanded_preds,
                                                          classes_to_plot=[0, 1], labels=['No Finding', 'Finding'],
                                                          interp_size=201, split_table=True),
        }, step=epoch)
        wandb.log({
            f'global/{val_test}_cm': wandb.plot.confusion_matrix(y_true=all_labels.squeeze().tolist(), probs=expanded_preds,
                                                                 class_names=['No Finding', 'Finding'], split_table=True),
        }, step=epoch)

    return es_metric_val


def log_metrics_to_wandb(per_client_metrics, global_round, test_set_key,
                         key_metric='auprc_macroavg', plot_per_client_metrics=True, plot_histogram=False):
    global_metrics_dict = {
        f'global/macro/{test_set_key}_{metric_name}': np.nanmean(np.array(
            [single_client_metrics[metric_name] for single_client_metrics in per_client_metrics.values()]))
        for metric_name in list(per_client_metrics.values())[0].keys()
    }
    for base_dataset_name in ['CXP', 'MDL']:
        if any([base_dataset_name in client_name for client_name in per_client_metrics.keys()]):
            global_metrics_dict = {**global_metrics_dict, **{
                f'{base_dataset_name}/macro/{test_set_key}_{metric_name}': np.nanmean(np.array(
                    [single_client_metrics[metric_name] for client_name, single_client_metrics in per_client_metrics.items()
                     if base_dataset_name in client_name]))
                for metric_name in list(per_client_metrics.values())[0].keys()
            }}
    wandb.log(global_metrics_dict, step=global_round)

    if plot_histogram:
        wandb.log({
           f'local/{test_set_key}_{metric_name}': wandb.Histogram([single_client_metrics[metric_name]
                                                        for single_client_metrics in per_client_metrics.values()
                                                        if not np.isnan(single_client_metrics[metric_name])])
           for metric_name in list(per_client_metrics.values())[0].keys()
        }, step=global_round)

    if plot_per_client_metrics:
        local_metrics_dict = {}
        for client_name, client_metrics in per_client_metrics.items():
            local_metrics_dict = {**local_metrics_dict, **{
                f'local/{test_set_key}_{metric_name}/{client_name}': metric_val
                for metric_name, metric_val in client_metrics.items()
            }}
        wandb.log(local_metrics_dict, step=global_round)

    return global_metrics_dict[f'global/macro/{test_set_key}_{key_metric}']
