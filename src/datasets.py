"""
This Python script contains implementations for:
- loading and combining various NLI datasets into a single dataset
- Pytorch Lightning Dataset and DataModule classes
"""

import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import sys
import numpy as np
import os
import copy
import random
from pathlib import Path
import requests


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def perturb_training_examples(dataset, fraction, random_state):

    def shuffle(text):
        text = ' '.join(random.sample(text.split(), len(text.split())))
        return text

    # 1. obtain random subset of dataset
    random_subset = dataset.sample(frac=fraction, random_state=random_state)

    # 2. apply perturbations (for now, just shuffling)
    random_subset['Premise'] = random_subset['Premise'].apply(shuffle)
    random_subset['Hypothesis'] = random_subset['Hypothesis'].apply(shuffle)

    # 3. append a '_perturbed' substring to the Dataset_id
    random_subset['Dataset'] = random_subset['Dataset'] + '_shuffled'

    # 4. delete sampled rows from clean dataset, then append perturbed subset to dataset.
    dataset.drop(random_subset.index, inplace=True)
    dataset = pd.concat([random_subset, dataset])

    print('NOTE: Shuffled training examples!', flush=True)

    return dataset


def downsample(dataset, dataset_id, downsample_rate, seed):
    """
    This function takes a dataframe and under samples it according to a provided percentage
    :return: under sampled dataframe
    """

    size_old = len(dataset)
    sample_size = int(downsample_rate * size_old)
    # print(f"{'Down-sampling {}% from {}:'.format(100*downsample_rate, dataset_id):<30}"
    #       f"{'from {} to {} examples'.format(size_old, sample_size):<32}", flush=True)

    return dataset.sample(sample_size, random_state=seed)


def undersample_to_smallest(df, seed):
    """
    Identifies smallest sub-dataset and undersamples all larger sub-datasets to match this dataset
    :param seed: random seed
    :param df: dataframe comprised of examples from multiple datasets. May be skewed towards 1 or 2 datasets.
    :return: dataframe
    """

    # determine number of examples in smallest sub dataset
    smallest_subdataset = df['Dataset'].value_counts().sort_values(ascending=True).index.tolist()[0]
    smallest_size = df['Dataset'].value_counts().sort_values(ascending=True).tolist()[0]

    subsets = []
    for key in df['Dataset'].unique():
        sub_dataset = df[df['Dataset'] == key]
        if len(sub_dataset) > smallest_size:
            print('Under-sampling {} to match size of minority sub-dataset {}'.format(key,
                                                                                      smallest_subdataset), flush=True)
        sub_df = sub_dataset.sample(smallest_size, random_state=seed)
        subsets.append(sub_df)

    df = pd.concat(subsets)
    return df


def apply_ratios(pool, dataset_ids, data_ratios, max_pool_size, seed):
    """
    This function takes a data pool and applies a provided list of ratios such that the pool is comprised of a
    particular combination over sources where ||pool|| = max_pool_size
    :param pool:
    :param dataset_ids:
    :param data_ratios:
    :param max_pool_size:
    :param seed:
    :return:
    """

    dataset_list = []

    for i in range(len(dataset_ids)):

        dataset_id = dataset_ids[i]
        fraction = data_ratios[i]
        num_samples = int(fraction * max_pool_size)
        sub_dataset = pool[pool['Dataset'] == dataset_id]
        dataset_list.append(sub_dataset.sample(num_samples, random_state=seed))

    new_pool = pd.concat(dataset_list, axis=0)

    print(new_pool.groupby('Dataset').count(), flush=True)
    print(new_pool.groupby('Label').count(), flush=True)

    return new_pool


def read_dataset(input_dir, dataset_id, split, seed, wanli_id, train_difficulty, config):
    """
    This function takes a dataset id and reads the corresponding .json split in an appropriate Pandas DataFrame
    :param wanli_id: id key for wanli json file (pairID for metrics, id for non metric stuff)
    :param seed:
    :param input_dir:
    :param dataset_id: str indicating which dataset should be read
    :param split: str indicating which split should be read
    :return: DataFrame with examples (Premise, Hypothesis, Label, ID)
    """
    # anli_rounds = ['ANLI_R1', 'ANLI_R2', 'ANLI_R3']
    anli_rounds = ['ANLI_R2', 'ANLI_R3']

    def replace_labels(label):
        label_conversions = {'e': 'entailment',
                             'c': 'contradiction',
                             'n': 'neutral'}

        return label_conversions[label]

    if dataset_id == 'SNLI':

        # TODO only consider snli with gold labels?
        data_path = '{}/snli_1.0/snli_1.0_{}.jsonl'.format(input_dir, split)
        dataset = pd.read_json(data_path, lines=True)
        dataset = dataset[['sentence1', 'sentence2', 'gold_label', 'pairID']]
        dataset['dataset'] = 'SNLI'
        dataset = dataset.drop(dataset[dataset.gold_label.str.contains('-')].index)  # drop examples with no gold label

    # load separate ANLI rounds
    elif dataset_id in anli_rounds:

        data_round = dataset_id[-2:]  # extract the 'R#' substring from the provided ID
        data_path = '{}/anli_v1.0/{}/{}.jsonl'.format(input_dir, data_round, split)
        dataset = pd.read_json(data_path, lines=True)
        dataset = dataset[['context', 'hypothesis', 'label', 'uid']]  # get rid of unnecessary columns
        dataset['label'] = dataset['label'].apply(replace_labels)  # ensures consistently named labels
        dataset['dataset'] = 'ANLI_{}'.format(data_round)
        #
        # print('Dataframe for {} - {}'.format(dataset_id, split), flush=True)

    # compile all ANLI rounds
    elif dataset_id == 'ANLI':

        dataset_list = []
        for data_round in anli_rounds:
            data_round = data_round[-2:]
            data_path = '{}/anli_v1.0/{}/{}.jsonl'.format(input_dir, data_round, split)
            dataset = pd.read_json(data_path, lines=True)
            dataset = dataset[['context', 'hypothesis', 'label', 'uid']]  # get rid of unnecessary columns
            dataset['label'] = dataset['label'].apply(replace_labels)  # ensures consistently named labels
            dataset['dataset'] = 'ANLI'
            dataset_list.append(dataset)
        dataset = pd.concat(dataset_list, axis=0)

    # Load MNLI
    elif dataset_id == 'MNLI':

        if split == 'train':
            data_path = '{}/multinli_1.0/multinli_1.0_{}.jsonl'.format(input_dir, split)
        elif split == 'dev' or split == 'test':
            data_path = '{}/multinli_1.0/multinli_1.0_dev_matched.jsonl'.format(input_dir)

        dataset = pd.read_json(data_path, lines=True)
        dataset = dataset[['sentence1', 'sentence2', 'gold_label', 'pairID']]
        dataset['dataset'] = 'MNLI'
        dataset = dataset.drop(dataset[dataset.gold_label.str.contains('-')].index)

        # split dev into dev and test set
        if split == 'dev':
            dataset = dataset.sample(frac=1,  # reshuffle all rows in the dataframe prior to split
                                     random_state=seed)  # seed is fixed such that dev and test set are always the same
            dataset = dataset[:int((len(dataset)/2))]

        elif split == 'test':
            dataset = dataset.sample(frac=1,
                                     random_state=seed)
            dataset = dataset[int((len(dataset)/2)):]

    # Load WANLI
    elif dataset_id == 'WANLI':

        if split == 'train':
            data_path = '{}/wanli/train.jsonl'.format(input_dir)
        elif split == 'dev' or split == 'test':
            data_path = '{}/wanli/test.jsonl'.format(input_dir)

        dataset = pd.read_json(data_path, lines=True)
        dataset = dataset[['premise', 'hypothesis', 'gold', wanli_id]]
        dataset['dataset'] = 'WANLI'
        dataset = dataset.drop(dataset[dataset.gold.str.contains('-')].index)

        # split dev into dev and test set
        if split == 'dev':
            dataset = dataset.sample(frac=1,  # reshuffle all rows in the dataframe prior to split
                                     random_state=seed)  # seed is fixed such that dev and test set are always the same
            dataset = dataset[:int((len(dataset) / 2))]

        elif split == 'test':
            dataset = dataset.sample(frac=1,
                                     random_state=seed)
            dataset = dataset[int((len(dataset) / 2)):]

    else:
        raise KeyError('No dataset found for "{}"'.format(dataset_id))

    # ensure consistent headers per dataset DataFrame
    dataset.columns = ['Premise', 'Hypothesis', 'Label', 'ID', 'Dataset']
    dataset['ID'] = dataset['ID'].astype('string')

    if split == 'train' and train_difficulty is not None:

        dataset = filter_by_difficulty(dataset, dataset_id, train_difficulty, seed, config)

    return dataset


def filter_by_difficulty(dataset, dataset_id, train_difficulty, seed, config):

    def load_confidences(path, predictions=False):

        df = pd.read_pickle(path)
        sample_ids = list(df.keys())

        cartography_dict = {}
        prediction_dict = {}

        for sample_id in sample_ids:

            times_correct = []
            confidences = []

            outputs = df[sample_id]
            outputs = outputs[1:]  # don't count the first 0.5 epoch!

            for output in outputs:
                # extract values from output tuple
                prediction = output[0]
                label = output[1]

                # obtain GT confidence, determine correctness of prediction
                confidence = prediction[label]
                correct = 1 if prediction.index(max(prediction)) == label else 0

                # track confidence and correctness
                confidences.append(confidence)
                times_correct.append(correct)

            cartography_dict[sample_id] = {'confidence': np.mean(confidences),
                                           'variability': np.std(confidences),
                                           'correctness': np.mean(times_correct)}

            prediction_dict[sample_id] = {'prediction': outputs[-1][0]}

        confidences = [cartography_dict[key]['confidence'] for key in list(cartography_dict.keys())]
        variability = [cartography_dict[key]['variability'] for key in list(cartography_dict.keys())]
        correctness = [cartography_dict[key]['correctness'] for key in list(cartography_dict.keys())]

        df = pd.DataFrame(list(zip(sample_ids, confidences, variability, correctness)),
                          columns=['id', 'mean_conf', 'variability', 'correctness'])

        if predictions == False:
            return df
        else:
            return df, prediction_dict

    # 1 load list of confidences for this seed;
    parent_dir = config.output_dir + '/datamaps/test/datamap_test_subepoch_inference_elke_0.5_epochs/' + config.model_id + '/random/' + str(config.seed)
    filepath = parent_dir + '/confidences.pickle'

    confidence_df = load_confidences(filepath)

    def get_difficulty_segment(confidences, dataset, difficulty, N=7000):

        # first select examples from particular segment
        if difficulty == 'easy':
            segment = confidences[confidences.mean_conf >= 0.75]
        elif difficulty == 'medium':
            segment = confidences[(confidences.mean_conf >= 0.5) & (confidences.mean_conf < 0.75)]
        elif difficulty == 'hard':
            segment = confidences[(confidences.mean_conf >= 0.25) & (confidences.mean_conf < 0.5)]
        elif difficulty == 'impossible':
            segment = confidences[(confidences.mean_conf < 0.25)]

        # then see which of those examples are in the current dataset and truncate to N examples per segment
        return dataset[dataset.ID.isin(segment.id)] #.sample(n=N, random_state=seed)

    algo = 'per_product'

    if algo == 'per_segment':
        dataset_list = []
        for difficulty in train_difficulty.split('_'):

            difficulty_segment = get_difficulty_segment(confidences=confidence_df,
                                                        dataset=dataset,
                                                        difficulty=difficulty)

            print('Number of examples in {}: {}'.format(difficulty, len(difficulty_segment)), flush=True)
            dataset_list.append(difficulty_segment)

        dataset = pd.concat(dataset_list)
        print(len(dataset), flush=True)

        return dataset

    elif algo == 'per_product':

        print('Current dataset size: {}'.format(len(dataset[dataset.ID.isin(confidence_df.id)])), flush=True)
        print('Filtering out bottom 25 percentage of data with lowest confidence-variability products', flush=True)
        confidence_df['product'] = confidence_df['mean_conf'] * confidence_df['variability']
        confidence_df = confidence_df.sort_values(by=['product'])[int(0.25 * len(confidence_df)):]
        print('Dataset size after filtering: {}'.format(len(dataset[dataset.ID.isin(confidence_df.id)])), flush=True)

        return dataset[dataset.ID.isin(confidence_df.id)]

def combine_datasets(input_dir,
                     datasets,
                     checkpoint_datasets,
                     dev_sets,
                     data_ratios,
                     max_pool_size,
                     max_dev_size,
                     split,
                     config,
                     downsample_rate,
                     seed,
                     wanli_id,
                     train_difficulty,
                     undersample,
                     perturb):
    """
    This function takes a list of NLI dataset names and
    concatenates all examples from each corresponding dataset
    for the provided data split (train/dev/test) into a single multi-dataset.

    :param exclude_from_training: any dataset in this list will be excluded when constructing the unlabelled train set
    :param checkpoint_datasets: any dataset in this list will be used for checkpointing. if None, dev sets are undersampled
    :param perturb:
    :param max_pool_size:
    :param data_ratios:
    :param undersample: flag to toggle under sampling w.r.t to minority dataset
    :param seed: random seed for reproducibility during sampling
    :param downsample_rate: value between 0 and 1, if we want to sample a subset of the data
    :param input_dir: SCRATCH dir where data files are read
    :param datasets: list with dataset names
    :param split: string indicating data split of interest
    :return: DataFrame with examples and labels for all datasets for split of interest
    """

    def id2index(dataframe):
        # reset index and assign to ID column
        dataframe = dataframe.reset_index(drop=True)
        dataframe['indices'] = dataframe.index
        return dataframe

    # If we consider multiple datasets we have to combine them into a single dataset
    # 1. create empty dataframe to store examples from all datasets
    dataset_list = []

    # 2. load individual datasets and append to list
    for i in range(len(datasets)):

        dataset_id = datasets[i]
        dataset = read_dataset(input_dir=input_dir,
                               dataset_id=dataset_id,
                               split=split,
                               seed=seed,
                               wanli_id=wanli_id,
                               train_difficulty=train_difficulty,
                               config=config)

        if split == 'train' and data_ratios != None and max_pool_size != None:

            fraction = data_ratios[i]
            sample_size = int(max_pool_size*fraction)
            print('Subsampling {} to size {}'.format(dataset_id, sample_size))
            dataset = dataset.sample(n=sample_size, random_state=seed)

        if split == 'dev' and checkpoint_datasets is None and max_dev_size != None and dataset_id in dev_sets:
            print('Running non AL experiment with chkpt based on aggregate dev performance.'
                  '\nDev set has to be made up from even sub-datasets. ANLI has smallest dev set (N=2200). '
                  '\nTherefore subsampling {} to size {}\n'.format(dataset_id, max_dev_size))
            dataset = dataset.sample(n=max_dev_size, random_state=seed)

        # 3. add dataset to multi-dataset
        dataset_list.append(dataset)

    # 4. combine individual datasets into single dataset
    combined_dataset = pd.concat(dataset_list, axis=0)

    # 4.1A for AL, under-sample larger sub-datasets to match smallest sub-dataset
    if max_pool_size is None:
        if undersample is True and len(datasets) > 1 and split == 'train':  # only applies to multi-data experiments
            combined_dataset = undersample_to_smallest(df=combined_dataset,
                                                       seed=seed)

        # Optional: down-sample training dataset:
        if 0 < downsample_rate < 1 and split == 'train':
            combined_dataset = combined_dataset.sample(frac=downsample_rate)

    # 4.2 Optional: add some perturbations to training set:
    if perturb is True and split == 'train':
        combined_dataset = perturb_training_examples(dataset=combined_dataset,
                                                     fraction=0.5,
                                                     random_state=seed)

    # 5. replace str ID with index-based ID system
    combined_dataset = id2index(combined_dataset)

    for dataset_id in combined_dataset['Dataset'].unique().tolist():
        sub_dataset = combined_dataset[combined_dataset['Dataset'] == dataset_id]
        print(f"{'{} {} size:'.format(dataset_id, split):<30}{len(sub_dataset):<32}", flush=True)

    print(f"{'Total {} size:'.format(split):<30}{len(combined_dataset):<32}", '\n', flush=True)

    return combined_dataset


class DataPool(Dataset):
    """
    This class implements the Pytorch Lightning Dataset object for multiple NLI datasets
    """

    def __init__(self, config, datasets, split, seed):
        """
        :param datasets: datasets: list with dataset names
        :param split: string indicating which split should be accessed
        :param model: string indicating which language model will be used for tokenization
        """

        # create single multi-dataset for desired data split (train, dev or test)
        # for the training split, we consider all examples as unlabelled and start training with a seed set L

        self.input_dir = config.input_dir
        self.datasets = datasets
        self.data_ratios = config.data_ratios
        self.max_pool_size = config.max_train_size
        self.max_dev_size = config.max_dev_size
        self.seed_size = config.seed_size
        self.downsample_rate = config.downsample_rate
        self.max_length = config.max_length
        self.model_id = config.model_id
        self.seed_datasets = config.seed_datasets
        self.random_seed = seed
        self.undersample = config.undersample
        self.perturb = config.perturb
        self.L = []
        self.config = config

        if split == 'train':

            data_ratios = []
            train_sets = []
            for dataset in datasets:
                ratio = float(dataset.split('_')[1])
                train_set = dataset.split('_')[0]
                train_sets.append(train_set)
                data_ratios.append(ratio)
            datasets = train_sets

            # first, we combine multiple NLI datasets into a single dataset and compile them in unlabeled pool U
            self.U = combine_datasets(input_dir=self.input_dir,
                                      datasets=datasets,
                                      data_ratios=data_ratios,
                                      checkpoint_datasets=config.checkpoint_datasets,
                                      dev_sets=config.dev_sets,
                                      max_pool_size=self.max_pool_size,
                                      max_dev_size=self.max_dev_size,
                                      split=split,
                                      downsample_rate=self.downsample_rate,
                                      seed=self.random_seed,
                                      config=self.config,
                                      wanli_id=self.config.wanli_id_key,
                                      train_difficulty = self.config.train_difficulty,
                                      undersample=self.undersample,
                                      perturb=self.perturb)

            # then, we label k samples randomly and put them in labeled pool L
            self.L = self.label_instances_randomly(k=self.seed_size,
                                                   dataset_ids=self.datasets)

            self.total_size = len(self.U) + len(self.L)

        else:

            # for dev and test we assume that all the data is labelled, so everything is passed to L
            self.L = combine_datasets(input_dir=self.input_dir,
                                      datasets=self.datasets,
                                      checkpoint_datasets=config.checkpoint_datasets,
                                      dev_sets=config.dev_sets,
                                      data_ratios=None,
                                      max_pool_size=None,
                                      max_dev_size=self.max_dev_size,
                                      split=split,
                                      downsample_rate=self.downsample_rate,
                                      seed=self.random_seed,
                                      config=self.config,
                                      wanli_id=self.config.wanli_id_key,
                                      train_difficulty=self.config.train_difficulty,
                                      undersample=self.undersample,
                                      perturb=self.perturb)

        self.data = self.L
        self.label2id = {"entailment": 0,
                         "contradiction": 1,
                         "neutral": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        def catch_index_error(index):
            try:
                self.data.iloc[index]
            except IndexError:
                print(index, flush=True)
                sys.exit()

        catch_index_error(index=idx)

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if len(self.data.iloc[idx]) > 6:
            print(self.data.columns, flush=True)
            print(self.data.iloc[idx], flush=True)

        premise, hypothesis, label, sample_id, dataset_id, sample_idx = self.data.iloc[idx]

        label = self.label2id[label]

        sample = {'premise': premise,
                  'hypothesis': hypothesis,
                  'label': label,
                  'sample_id': sample_id,
                  'dataset_id': dataset_id}

        return sample

    def set_mode(self, mode):
        """
        Calling this function sets self.data to point to:
        - the labelled set, if we seek to access the labelled examples in L
        - the unlabelled set, if we seek to access the unlabelled examples U
        :param mode:
        :return:
        """

        if mode == 'L':
            self.data = self.L

        elif mode == 'U':
            self.data = self.U

        else:
            raise KeyError('{} is not a valid mode. Use mode=L or mode=U.'.format(mode))

    def set_k(self, k):
        """
        if k is an integer we draw k samples from the unlabelled pool
        if k is real-valued between 0 and 1 we draw the corresponding % from the unlabelled pool
        """

        if k == 0:
            print('Warning: value of k is 0 - this means no samples will be acquired for labelling!')

        elif 0 < k < 1:
            k = int(k * (len(self.U) + len(self.L)))

        if k == 1.0:
            k = int(k * (len(self.U) + len(self.L)))

        if k > 1:
            pass

        return k

    def save_example_ids(self, new_examples, active_round):

        df_copy = new_examples.copy()

        # save IDs corresponding to indices in run-specific dir
        parent_dir = self.config.output_dir + '/acquisition_IDs/' + self.config.project_dir + '/' + self.config.array_uid.replace(' ','_') + '/' + self.config.model_id + '/' + self.config.acquisition_fn + '/' + str(self.config.seed)
        filepath = parent_dir + '/acquisition_ids.csv'
        df_copy['round'] = active_round  # add an identifier for the round we're currently in

        if os.path.isfile(filepath):
            print('Appending sample IDs to existing .csv', flush=True)
            df_copy[['round', 'ID', 'Dataset', 'Label']].to_csv(filepath, mode='a', index=False, header=False)  # append to existing file
        else:
            print('Creating new dir to write .csv', flush=True)
            Path(parent_dir).mkdir(parents=True, exist_ok=True)
            df_copy[['round', 'ID', 'Dataset', 'Label']].to_csv(filepath, index=False)  # write to new file
        return None

    def label_instances_randomly(self, k, dataset_ids):
        """
        This function randomly selects k examples from the unlabelled set U
        and transfers them to the labelled set L
        :param dataset_ids:
        :param k: size of initial pool of labelled examples
        :return:
        """

        # select k instances from U to be labelled for initial seed L.
        # Make sure to remove these instances from U.
        k = self.set_k(k)  # check if provided k is a percentage or an integer
        self.U = self.U.reset_index(drop=True)

        print('Drawing {} random samples from {} from unlabelled set U for labelled seed set L'.format(k, self.U['Dataset'].unique()),
              flush=True)

        # initialize empty seed dataset L
        L = pd.DataFrame(columns=self.U.columns)

        # From the unlabelled pool, consider all examples from the datasets that we want to use for the seed;
        # Then, sample k examples from this subset and use the original index values to extract them from U via .iloc
        random_indices = self.U.sample(k, random_state=self.random_seed).index.values
        labelled_examples = self.U.iloc[random_indices]

        # write example IDs corresponding to indices of current round to file
        self.save_example_ids(new_examples=labelled_examples, active_round=0)

        L = L.append(labelled_examples).reset_index(drop=True)
        self.U = self.U.drop(labelled_examples.index).reset_index(drop=True)
        print(f"{'Total size unlabelled pool:':<30}{len(self.U):<32}")
        print(f"{'Total size labelled pool:':<30}{len(L):<32}")

        return L

    def label_instances(self, indices, active_round, save_ids=True):
        """
        This function takes an array of indices and transfers the corresponding examples from the unlabelled pool U
        to the labelled pool L
        :param save_ids: toggle saving of acquired IDs. true by default, False should be passed when computing metrics
        :param active_round: specifies which iteration of AL we're currently in
        :param indices: np array with indices of samples that we want to label based on some acquisition function
        :return:
        """

        if len(indices) > len(self.U):
            print('More indices gathered than remaining unlabelled examples. This should not happen!', flush=True)
            indices = indices[:len(self.U)]

        new_examples = self.U.iloc[indices]  # Take examples from unlabelled pool
        if save_ids:
            self.save_example_ids(new_examples=new_examples, active_round=active_round)  # save IDS corresponding to indices for this round

        self.L = self.L.append(new_examples).reset_index(drop=True)  # Add them to the labelled pool
        self.U = self.U.drop(new_examples.index).reset_index(drop=True)  # Remove examples from unlabelled pool
        self.assert_validity()  # check whether U and L are disjoint and whether all indices are unique
        print('Labelled {} new instances'.format(len(new_examples)), flush=True)
        print(f"{'Total size unlabelled pool:':<30}{len(self.U):<32}", flush=True)
        print(f"{'Total size labelled pool:':<30}{len(self.L):<32}", flush=True)

        print('Finished labelling \n')

        return None

    def assert_validity(self):
        """
        re-usable function for ensuring that U and L do not overlap
        """
        # assert that U and L are disjoint
        unlabelled_indices = set(self.U.indices.tolist())
        labelled_indices = set(self.L.indices.tolist())
        assert unlabelled_indices.isdisjoint(labelled_indices)

        # assert that both U and L have only unique indices
        assert len(self.U) == len(set(self.U.indices.tolist()))
        assert len(self.L) == len(set(self.L.indices.tolist()))

        # assert that both datasets still add up to original no. of examples
        assert len(self.U) + len(self.L) == self.total_size
        return None

    def load_past_indices(self, min_round, relative_dataset):
        """
        This function does the following:
        1. loads the acquisitions_ID file corresponding to some experiment ID and a specific seed and acquisition fn
        2. selects the acquired data from U using the IDs, returns the indices
        :return:
        """

        # 1. load acquired data as dm
        acquired_df = self.load_past_acquired_data()
        acquired_df['round'] = acquired_df['round'].astype('int')
        # 2. remove the rows for round 1 since those are already in L; for the remainder select only the IDs column
        print('Loading all data between rounds {} and 14'.format(min_round), flush=True)
        acquired_df_ids = acquired_df[(acquired_df['round'] >= min_round) & (acquired_df['round'] < 15)]['ID']

        # convert ID dtype to str
        acquired_df_ids = acquired_df_ids.astype('string').tolist()
        relative_dataset.ID = relative_dataset.ID.astype('string')

        # 3. use IDs to select data points from unlabelled set; then get the indices of those examples
        indices = []
        seen = []

        for i, example in enumerate(relative_dataset.ID.tolist()):
            if example in acquired_df_ids and example not in seen:
                indices.append(i)
            seen.append(example)

        print('No of indices found: ', flush=True)
        print(len(indices), flush=True)

        print('{} duplicates found in acquired data'.format(len(seen)), flush=True)
        indices = list(set(indices))

        return indices

    def load_past_acquired_data(self):

        # load acquired data
        parent_dir = self.config.output_dir + '/acquisition_IDs/' + self.config.project_dir + '/' + self.config.metrics_uid.replace(
            ' ', '_') + '/' + self.config.model_id + '/' + self.config.acquisition_fn + '/' + str(self.config.seed)
        filepath = parent_dir + '/acquisition_ids.csv'

        acquired_df = pd.read_csv(filepath)

        return acquired_df


def get_tokenizer(model_id):

    done = False
    while done is False:
        try:

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            done = True

        except requests.exceptions.RequestException as exception:
            print('Got internal server error. Trying to download tokenizer again in 10 seconds..', flush=True)
            print(exception)
            time.sleep(10)

    return tokenizer

class GenericDataModule(pl.LightningDataModule):
    """
    This Lightning module produces DataLoaders using DataPool instances
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tokenizer = get_tokenizer(model_id=config.model_id)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer,
                                                padding='longest',
                                                max_length=config.max_length)

        self.train = None
        self.val = None
        self.test = None
        self.label2id = {"entailment": 0,
                         "contradiction": 1,
                         "neutral": 2}

        self.pin_memory = True if self.config.gpus > 0 else False

    def batch_tokenize(self, batch):

        premises = [sample['premise'] for sample in batch]
        hypotheses = [sample['hypothesis'] for sample in batch]
        labels = [sample['label'] for sample in batch]
        ids = [sample['sample_id'] for sample in batch]
        dataset_ids = [sample['dataset_id'] for sample in batch]

        labels = torch.tensor(labels)

        # tokenize sentence and convert to sequence of ids
        tokenized_input_seq_pairs = self.tokenizer.__call__(text=premises,
                                                            text_pair=hypotheses,
                                                            add_special_tokens=True,
                                                            max_length=350,
                                                            padding='longest',
                                                            return_attention_mask=True,
                                                            return_token_type_ids=True,
                                                            return_tensors='pt',
                                                            truncation=True)

        input_ids = tokenized_input_seq_pairs['input_ids']
        token_type_ids = tokenized_input_seq_pairs['token_type_ids']
        attention_masks = tokenized_input_seq_pairs['attention_mask']

        # TODO: check whether batches are actually of variable seq length and not 350 on average

        padded_batch = {'input_ids': input_ids,
                        'token_type_ids': token_type_ids,
                        'attention_masks': attention_masks,
                        'labels': labels,
                        'sample_ids': ids,
                        'dataset_ids': dataset_ids}

        return padded_batch

    def setup(self, stage=None):

        fixed_seed = self.config.seed

        if stage == 'fit':

            print('\nBuilding train pool..', flush=True)
            if self.train is None:

                self.train = DataPool(config=self.config,
                                      datasets=self.config.train_sets,
                                      split='train',
                                      seed=fixed_seed)

            print('\nBuilding dev and test sets..', flush=True)
            if self.val is None:
                dev_split = 'dev'
                if self.config.test_datamap:
                    dev_split = 'test'
                self.val = DataPool(config=self.config,
                                    datasets=self.config.dev_sets,
                                    split=dev_split,
                                    seed=fixed_seed)

            if self.config.ood_sets is not None:
                print('Building OOD dev set...', flush=True)
                self.ood_val = DataPool(config=self.config,
                                        datasets=self.config.ood_sets,
                                        split='dev',
                                        seed=fixed_seed)

            if self.test is None:
                # self.config.seed = 39
                self.test = DataPool(config=self.config,
                                     datasets=self.config.test_sets,
                                     split='test',
                                     seed=fixed_seed)

        if stage == 'test':
            pass

        print('Done building datasets!', flush=True)

    def train_dataloader(self):

        return DataLoader(self.train,
                          collate_fn=self.batch_tokenize,
                          shuffle=True,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=False)

    def val_dataloader(self):

        return DataLoader(self.val,
                          collate_fn=self.batch_tokenize,
                          shuffle=False,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=False)

    def test_dataloader(self):

        return DataLoader(self.test,
                          shuffle=False,
                          collate_fn=self.batch_tokenize,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=False)

    def labelled_dataloader(self, shuffle=True):

        self.train.set_mode('L')
        return DataLoader(self.train,
                          collate_fn=self.batch_tokenize,
                          shuffle=shuffle,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=False)

    def unlabelled_dataloader(self, batch_size=None, shuffle=False):

        batch_size = self.config.batch_size if batch_size is None else batch_size
        self.train.set_mode('U')
        return DataLoader(self.train,
                          collate_fn=self.batch_tokenize,
                          shuffle=shuffle,
                          batch_size=batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=False)

    def separate_loader(self, sub_datapool):

        return DataLoader(dataset=sub_datapool,
                          collate_fn=self.batch_tokenize,
                          shuffle=False,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=False)

    def get_separate_loaders(self, split, dataset_ids):

        if split == 'dev':
            aggregate_datapool = self.val
        elif split == 'test':
            aggregate_datapool = self.test
        elif split == 'ood':
            aggregate_datapool = self.ood_val
        else:
            raise KeyError('Please specify for which split datasets should be separated')

        separate_loaders = []
        for dataset_id in dataset_ids:  # for each dataset seen during training
            sub_datapool = copy.deepcopy(aggregate_datapool)  # make a clone of the DataPool test-set object,
            sub_datapool.data = sub_datapool.L[sub_datapool.L['Dataset'] == dataset_id]  # filter based on dataset ID

            sub_dataloader = self.separate_loader(sub_datapool=sub_datapool)  # create dataloader for subset
            separate_loaders.append(sub_dataloader)

        return separate_loaders

    def has_unlabelled_data(self):
        return len(self.train.U) > 0

    def has_labelled_data(self):
        return len(self.train.L) > 0

    def label(self, indices):
        self.train.label_indices(indices)

    def dump_csv(self):

        self.train.U.to_csv(path_or_buf='{}/U_pool_seed_{}.csv'.format(self.config.output_dir, self.config.seed))
        self.train.L.to_csv(path_or_buf='{}/L_pool_seed_{}.csv'.format(self.config.output_dir, self.config.seed))
        return None