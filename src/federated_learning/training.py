import logging
import time
import copy

import torch
import wandb
import numpy as np

from src.federated_learning.evaluation import evaluate_model
from src.trainer import Trainer
from src.utils.io_utils import check_path
from src.utils.early_stopping import EarlyStopping


def run_federated_learning(global_model, clients, use_gpu, cfg):
    fed_start = time.time()
    best_es_val = 0
    track_no_improv = 0
    client_pool = clients  # clients that may be selected for training

    # Set up wandb table for per-client performances
    header = ['client_name', 'local_epoch', 'train_time', 'train_loss',
              'val_loss', 'val_auroc', 'val_auprc', 'val_auprc_macroavg', 'val_f1', 'val_accuracy', 'val_mcc']
    if cfg.training.track_norm:
        header += ['track_norm']
    if cfg.dp.enabled:
        header += ['epsilon', 'delta', 'best_alpha']
    metrics_table = wandb.Table(columns=header)

    early_stopper = EarlyStopping('macro_mcc', 0, cfg.training.early_stopping_rounds, 'max')

    def _sample_clients():
        x = np.random.uniform(size=len(client_pool))
        sampled_clients = [
            client_pool[i] for i in range(len(client_pool))
            if x[i] < cfg.training.q]
        if len(sampled_clients) == 0:
            sampled_clients = [np.random.choice(client_pool)]

        return sampled_clients

    for global_round in range(cfg.training.T):
        logging.info(f"[[[ Round {global_round} Start ]]]")
        train_start_time = time.time()

        # Step 1: select random fraction of clients
        sel_clients = _sample_clients()
        for sel_client in sel_clients:
            sel_client.selected_rounds += 1

        # check if clients have now exceeded the maximum number of rounds they can be selected
        # and drop them from the pool if so
        for cp in client_pool:
            if cp.selected_rounds == cfg.training.max_client_selections:
                client_pool.remove(cp)
        logging.info(f"Number of selected clients: {len(sel_clients)}")
        logging.debug(f"Clients selected: {[sel_cl.name for sel_cl in sel_clients]}")

        # Step 2: send global model to clients and train locally
        for client_k in sel_clients:
            # reset model at client's site
            client_k.model_params = None
            # https://blog.openmined.org/pysyft-opacus-federated-learning-with-differential-privacy/
            with torch.no_grad():
                for client_params, global_params in zip(client_k.model.parameters(), global_model.parameters()):
                    client_params.set_(copy.deepcopy(global_params))

            logging.info(f"<< {client_k.name} Training Start >>")
            if cfg.save_model:
                # set output path for storing models and results
                client_k.output_path = cfg.output_path + f"round{global_round}_{client_k.name}/"
                client_k.output_path = check_path(client_k.output_path, warn_exists=False)

            # Step 3: Perform local computations
            # returns local best model
            train_valid_start = time.time()
            train_metrics = Trainer.train(client_k, cfg, use_gpu, out_csv=f"round{global_round}_{client_k.name}.csv",
                                          freeze_mode=cfg.model.freeze_layers)
            for i in range(len(train_metrics[0])):
                # calculate running local epoch
                single_epoch_metrics = [metric[i] for metric in train_metrics]
                single_epoch_metrics[0] = global_round * cfg.training.E + single_epoch_metrics[0]
                metrics_table.add_data(client_k.name, *single_epoch_metrics)
            client_k.model_params = client_k.model.state_dict().copy()
            client_time = round(time.time() - train_valid_start)
            logging.debug(f"<< {client_k.name} Training Time: {client_time} seconds >>")

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
            first_cl.model_params[key] = sum(weights) / sum(weight_n)  # weighted averaging model weights

        # Step 5: server updates global state
        global_model.load_state_dict(first_cl.model_params)

        if cfg.save_model:
            # also save intermediate models
            torch.save(global_model.state_dict(), cfg.output_path + f"global_{global_round}rounds.pth.tar")
        wandb.log({'durations/training': time.time() - train_start_time}, step=global_round)

        # Step 6: validate and early stopping
        logging.info("Validating global model...")
        validation_start_time = time.time()

        es_val = evaluate_model(global_model, clients, global_round, 'val', use_gpu, cfg, plot_curves=False)

        # track early stopping & lr decay
        if es_val > best_es_val:
            best_es_val = es_val
            track_no_improv = 0
        else:
            track_no_improv += 1
            if track_no_improv == cfg.training.reduce_lr_rounds:
                # decay lr
                cfg.training.lr = cfg.training.lr * 0.1
                logging.info(f"Learning rate reduced to {cfg.training.lr}")
            elif track_no_improv == cfg.training.early_stopping_rounds:
                logging.info(f'Global AUC has not improved for {track_no_improv} rounds. Stopping training.')
                break
        wandb.log({'durations/validation': time.time() - validation_start_time}, step=global_round)

        logging.info(f"[[[ Round {global_round} End ]]]\n")

    logging.info("Global model trained")
    fed_end = time.time()
    logging.info(f"Total training time: {round(fed_end - fed_start)}")

    # Log per-client metrics
    wandb.log({'client_metrics_table': metrics_table}, step=global_round)
    #for metric in ['auroc', 'auprc', 'f1']:
    #    wandb.log({f"client_{metric}s": wandb.plot_table(
    #        f"wandb/lineseries/v0", metrics_table, {"step": "local_epoch",
    #                                                "lineVal": f'val_{metric}',
    #                                                "lineKey": "client_name",
    #                                                "xname": "Local Epoch"},
    #        {"title": f"Client {metric}s"})
    #    }, step=global_round)

    return global_model
