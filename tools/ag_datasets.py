import os
import numpy as np
import pandas as pd
import math
import sys


from statistics import mean
from statistics import pstdev


def build_audiogram_dataset(data_path,
                            use_hermes=True,
                            use_wusmci=True,
                            use_contralateral=True,
                            replace_invalid=True,
                            force_dtype='float32',
                            clip=None):

    component_datasets = {}

    if use_hermes:

        # define filepath
        FILE_HERMES = 'HermesPost.xlsx'
        PATH_HERMES = os.path.join(data_path, FILE_HERMES)

        # Define column headers to use from each raw dataset
        hermes_headers_ip = [
            'ia125hz', 'ia250hz', 'ia500hz', 'ia750hz', 'ia1000hz', 'ia1500hz',
            'ia2000hz', 'ia3000hz', 'ia4000hz', 'ia6000hz', 'ia8000hz'
        ]

        hermes_headers_cl = [
            'contrabl125hz', 'contrabl250hz', 'contrabl500hz', 'contrabl750hz',
            'contrabl1000hz', 'contrabl1500hz', 'contrabl2000hz',
            'contrabl3000hz', 'contrabl4000hz', 'contrabl6000hz',
            'contrabl8000hz'
        ]

        if use_contralateral:
            component_datasets[PATH_HERMES] = [hermes_headers_ip, hermes_headers_cl]
        else:
            component_datasets[PATH_HERMES] = [hermes_headers_ip]

    if use_wusmci:

        FILE_WUSMCI = 'WUSM_CI_candidacy_dataset.xlsx'
        PATH_WUSMCI = os.path.join(data_path, FILE_WUSMCI)

        wusmci_headers_r = [
            'R 125Hz', 'R 250 Hz', 'R 500 Hz', 'R 750 Hz', 'R 1000 Hz',
            'R 1500 Hz', 'R 2000 Hz', 'R 3000 Hz', 'R 4000 Hz', 'R 6000 Hz',
            'R 8000 Hz'
        ]

        wusmci_headers_l = [
            'L 125 Hz', 'L 250 Hz', 'L 500 Hz', 'L 750 Hz', 'L 1000 Hz',
            'L 1500 Hz', 'L 2000 Hz', 'L 3000 Hz', 'L 4000 Hz', 'L 6000 Hz',
            'L 8000 Hz'
        ]

        if use_contralateral:
            component_datasets[PATH_WUSMCI] = [wusmci_headers_r, wusmci_headers_l]
        else:
            component_datasets[PATH_WUSMCI] = [wusmci_headers_r]

    if replace_invalid:
        replace_dict = {
            'NR': 125.0,
            'DNT': np.nan,
            'NT': np.nan,
            'dnt': np.nan,
            '45]0': 45.0,
            '65 VT': np.nan,
            '90 VT': np.nan,
            '12585': np.nan,
            'na': np.nan,
            '6570': np.nan,
            'N': np.nan,
            'NA\n': np.nan,
            'MA': np.nan,
            'nr': 125.0,
            '8-': np.nan,
            '110 NR': 125.0,
            'NR125': 125.0,
            '9090': 90.0
        }

    else:
        replace_dict = None

    dataset_headers = [
        '125hz', '250hz', '500hz', '750hz', '1000hz', '1500hz', '2000hz',
        '3000hz', '4000hz', '6000hz', '8000hz'
    ]

    df = build_dataset(dataset_headers=dataset_headers,
                       component_datasets=component_datasets,
                       replace_dict=replace_dict,
                       force_dtype=force_dtype)

    if clip:
        df = df.clip(lower=clip[0], upper=clip[1])

    return df


# General dataset construction function
# Takes in component datasets with specified headers and concatenates them row-wise into one
# larger dataset. Example structure for component_datasets below:
#    component_datasets = {
#       PATH_HERMES : [hermes_headers_ip, hermes_headers_cl],
#       PATH_WUSMCI : [wusmci_headers_r, wusmci_headers_l]
#   }
def build_dataset(dataset_headers,
                  component_datasets,
                  replace_dict=None,
                  force_dtype=None):

    df = pd.DataFrame()

    for datapath in component_datasets:

        raw_data = pd.read_excel(datapath)

        for headers in component_datasets[datapath]:
            subset = raw_data[headers]

            subset.columns = dataset_headers

            if df.empty:
                df = subset

            else:
                df = pd.concat([df, subset], ignore_index=True)

    if replace_dict:
        df.replace(replace_dict, inplace=True)

    if force_dtype:
        df = df.astype(force_dtype)

    return df


def find_parent_frequency(dataset, min_col=1, verbose=0):

    # Remove indices (patients) from the dataset who do not have a specified number of datapoints present
    parent_dataset = dataset.dropna(thresh=min_col)

    # find total number of patients present who have at least min_col features present
    total = parent_dataset.dropna(thresh=min_col).shape[0]
    num_drop_weights = []

    # Find the proportions of indices (patients) with X datapoints present
    for num_drop in range(min_col, parent_dataset.shape[1] + 1):
        count = parent_dataset.dropna(thresh=num_drop).shape[0]
        count_next = parent_dataset.dropna(thresh=num_drop + 1).shape[0]
        num_pts = count - count_next
        num_drop_weights.append(num_pts / total)

    # Reverse the list of proportions calculated above to find the relative proportions of indices (patients) with X datapoints dropped
    num_drop_weights.reverse()

    col_drop_weights = []

    # Find the relative proportions of points dropped per column
    for freq in range(0, parent_dataset.shape[1]):
        num_absent = parent_dataset.iloc[:, freq].isna().sum()
        col_drop_weights.append(num_absent / total)

    if verbose > 1:
        print('Proportion of patients with X datapoints dropped: ' + str(num_drop_weights))
        print('Proportion of datapoints dropped per feature: ' + str(col_drop_weights))

    return num_drop_weights, col_drop_weights

def drop_n(X, parent, dist_type, rate, drop_max, drop_proportion, verbose=0, min_col=1):

    num_drop_weights, col_drop_weights = find_parent_frequency(parent, min_col, verbose)

    # Drop datapoints from X pseudo-randomly, weighting number dropped per patient and per frequency
    for index in range(0, X.shape[0]):
        instance = X.iloc[index]

        # only apply drop-function to the specified proportion of instances
        drop_percent = drop_proportion * 100

        if np.random.randint(1, 100) <= drop_percent:

            if rate == 'parent':

                num_drop_weights_nozero = np.copy(num_drop_weights)
                num_drop_weights_nozero[0] = 0.0

                if drop_max is not None:
                    minimum_features_present = X.shape[1] - drop_max
                    num_drop_weights_nozero[(minimum_features_present+1):] = 0.0

                weight_sum = num_drop_weights_nozero.sum()

                num_drop_weights_nozero = [x/weight_sum for x in num_drop_weights_nozero]

                number_to_drop = np.random.choice(list(range(0, X.shape[1])), p=num_drop_weights_nozero)


            elif rate == 'random':
                if drop_max is None:
                    drop_max = X.shape[1] - 1

                number_to_drop = np.random.randint(1, drop_max+1)

            else:
                number_to_drop = rate

            # If some of the drop weights are 0, set to 1e-10 because pandas sample doesn't accept 0's
            col_drop_weights = np.clip(a=col_drop_weights, a_min=1e-10, a_max=1)

            # weight frequencies dropped using weights calculated above from parent dataset
            if dist_type == 'parent':
                instance[instance.sample(n=number_to_drop, weights=col_drop_weights).index] = np.nan

            elif dist_type == 'parent-inverse':



                col_drop_weights_i = [1 - x for x in col_drop_weights]

                # weight_sum = col_drop_weights_i.sum()
                # col_drop_weights_i = [x/weight_sum for x in col_drop_weights_i]
                instance[instance.sample(n=number_to_drop, weights=col_drop_weights_i).index] = np.nan


            elif dist_type == 'skew-terminal':

                col_drop_weights = np.ones(X.shape[1])
                col_drop_weights[0:3] *= 3
                col_drop_weights[-3:] *= 3

                weight_sum = col_drop_weights.sum()
                col_drop_weights = [x/weight_sum for x in col_drop_weights]
                instance[instance.sample(n=number_to_drop, weights=col_drop_weights).index] = np.nan

            elif dist_type == 'skew-central':
                col_drop_weights = np.ones(X.shape[1])
                col_drop_weights[3:-3] *= 3

                weight_sum = col_drop_weights.sum()
                col_drop_weights = [x/weight_sum for x in col_drop_weights]
                instance[instance.sample(n=number_to_drop, weights=col_drop_weights).index] = np.nan


            elif dist_type == 'random':
                instance[instance.sample(n=number_to_drop).index] = np.nan

            else:
                print('Invalid dist_type of: ', dist_type)

            X.iloc[index] = instance

    return X


def generate_sparse_dataset(
    parent,
    rate,
    dist_type,
    drop_proportion,
    drop_max=None,
    drop_cols=None,        # for 'custom' drop_dist - custom not currently implemented
    min_col=1,          # for "parent" drop_dist
    cols_to_ignore=None,
    verbose=0,
    size=None,
    prop=None
):
    '''
    Returns X, y, missing_mask
    drop_rate: for parent, if -1, drop num will mirror parent frequency
    '''

    if cols_to_ignore is not None:
        parent = parent.drop(columns=cols_to_ignore)

    # Initialize y using only indices for which ground truth data exists for all features in the dataset
    y = parent.dropna(thresh=parent.shape[1])

    if size is not None:


        size = int(size * prop)

        if size > y.shape[0]:
            size = y.shape[0]

        #print('size is ', size, ' and y shape is: ', y.shape)

        y = y.sample(n=size)

    # Initialize X as a copy of y
    X = y.copy()

    X = drop_n(X, parent, dist_type, rate, drop_max, drop_proportion, verbose=verbose, min_col=min_col)

    # Create a mask for all the values that were randomly dropped from the training dataset
    missing_mask = X.isna().to_numpy()

    return X, y, missing_mask
