import sys
import yaml
from datetime import datetime

from hydra_plugins.hydra_wandb_sweeper._impl import create_wandb_param_from_config


if __name__ == '__main__':
    experiment = sys.argv[1]
    with open(f'./config/experiment/{experiment}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    params = config['hydra']['sweeper']['params']
    params = {
        str(x): create_wandb_param_from_config(y, 'grid')
        for x, y in params.items()
    }

    with open('sweep_config.yaml', 'w') as file:
        sweep_config = {
            "program": "train_FL.py",
            "command": [
                "${env}",
                "${interpreter}",
                #"-m",
                "${program}",
                f"+experiment={experiment}",
                "${args_no_hyphens}"
            ],
            "early_terminate": {
                "eta": 2,
                "min_iter": 4,
                "type": "hyperband"
            },
            "method": "bayes",
            "metric": {
                "goal": "maximize",
                "name": "global/macro/test_mcc",
                "target": 1.0
            },
            "run_cap": 150,
            "name": f"{config['experiment_name']}_{datetime.now().strftime('%d-%m_%H-%M')}",
            "parameters": params
        }

        # parameters:
        yaml.dump(sweep_config, file, sort_keys=False)
