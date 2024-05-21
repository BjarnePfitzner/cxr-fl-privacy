from opacus.privacy_engine import get_noise_multiplier, DEFAULT_ALPHAS
import pandas as pd


def get_delta_for_n_data(n_data, max_delta):
    delta = min(max_delta, 1 / n_data * 0.1)

    # Convert the number to a string
    delta_str = str(delta)

    if 'e' in delta_str:
        decimal_index = int(delta_str[delta_str.find('-') + 1:])
        rounded_delta = round(delta, decimal_index)
    else:
        # Find the index of the first non-zero decimal digit
        decimal_index = delta_str.find('.') + 1
        while decimal_index < len(delta_str) and delta_str[decimal_index] == '0':
            decimal_index += 1

        # Round the number to that decimal place
        rounded_delta = round(delta, decimal_index - 1)

    return rounded_delta


EPSILONS = [1, 3, 6, 10]
MIN_DELTA = 1e-3
BATCH_SIZES = [10]#, 64, 128]
MAX_LOCAL_EPOCHS = 10

N_CHEXPERT_CLIENTS = 22
CHEXPERT_CLIENT_DATA = './data/chexpert_clients/'

N_MENDELEY_CLIENTS = 20
MENDELEY_CLIENT_DATA = './data/mendeley_clients/'

print('\\toprule')
print('Base & \\multirow{2}{*}{||x_k||_1} & \\multirow{2}{*}{\\delta} & \\multicolumn{' + str(len(EPSILONS)) + '}{c}{\\varepsilon} \\\\')
print('Dataset & & ' + ' & '.join([str(eps) for eps in EPSILONS]) + ' \\\\')
print('\\midrule')
for i in range(N_CHEXPERT_CLIENTS + N_MENDELEY_CLIENTS):
    if i == 0:
        row_str = '\\multirow{' + str(N_CHEXPERT_CLIENTS) + '}{*}{\\rotatebox{90}{CheXpert}} & '
    elif i == N_CHEXPERT_CLIENTS:
        print('\\midrule')
        row_str = '\\multirow{' + str(N_CHEXPERT_CLIENTS) + '}{*}{\\rotatebox{90}{Mendeley}} & '
    else:
        row_str = '& '
    client_id = i % N_CHEXPERT_CLIENTS
    if client_id == i:
        # CheXpert client
        base_path = CHEXPERT_CLIENT_DATA
    else:
        # Mendeley client
        base_path = MENDELEY_CLIENT_DATA
    client_data = pd.read_csv(f'{base_path}client_{client_id}/client_train.csv')
    base_n_data = len(client_data)
    row_str += f'{str(base_n_data)} & '

    delta = get_delta_for_n_data(base_n_data, MIN_DELTA)
    delta_str = f'{delta:.0e}'.replace('e', ' \\times 10^{') + '}'
    row_str += f'{delta_str} & '
    for epsilon in EPSILONS:
        for batch_size in BATCH_SIZES:
            if base_n_data > batch_size:
                n_data = base_n_data - base_n_data % batch_size
            else:
                n_data = base_n_data
            z = get_noise_multiplier(target_epsilon=epsilon,
                                     target_delta=delta,
                                     sample_rate=min(1.0, batch_size/n_data),
                                     epochs=MAX_LOCAL_EPOCHS,
                                     alphas=DEFAULT_ALPHAS)
            row_str += f'{str(round(z, 2))} & '

    row_str = row_str[:-2] + '\\\\'
    print(row_str)
print('\\bottomrule')