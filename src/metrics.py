import torch
import copy
import numpy as np
import os
from pathlib import Path

from src.utils import get_trainer, train_model, del_checkpoint, get_model, evaluation_check
from src.datasets import GenericDataModule
from statistics import mean
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import gc


class Metrics:

    def __init__(self, config, train_logger, metric_logger):

        print('\nCOMPUTING METRICS\n', flush=True)
        self.config = config
        self.train_logger = train_logger
        self.metric_logger = metric_logger
        self.dm = None

        pass

    def init_dm(self):

        print('Reinitialising data module..', flush=True)
        self.dm = None
        self.dm = GenericDataModule(config=self.config)
        self.dm.setup(stage='fit')

        return None

    def log(self, metric):

        self.metric_logger.log(metric)

        return None

    def compute_metric(self, metric):

        if metric == 'uncertainty':
            self.compute_uncertainty()

        elif metric == 'input_diversity':
            self.compute_input_diversity()

        elif metric == 'feature_diversity':
            self.compute_feature_diversity()

        else:
            raise ValueError('"{}" Is not a valid option! \nChoose from ["uncertainty", "input_diversity", "feature_diversity"]'.format(metric))

        return None

    def _train_model(self):

        all_results = []

        done = False
        while done is False:

            model = get_model(config=self.config,
                              train_loader=self.dm.labelled_dataloader())

            trainer = get_trainer(config=self.config,
                                  logger=self.train_logger)

            model, dm, logger, trainer = train_model(dm=self.dm,
                                                     config=self.config,
                                                     model=model,
                                                     logger=self.train_logger,
                                                     trainer=trainer)

            # check whether model ran successfully
            failure, validation_score = evaluation_check(dm=self.dm,
                                                         config=self.config,
                                                         model=model,
                                                         trainer=trainer,
                                                         current_results=all_results)

            del_checkpoint(trainer.checkpoint_callback.best_model_path)

            if failure:
                print('Model failed to learn. Restarting training for this AL iteration with a different seed..',
                      flush=True)
                del model
                del trainer
                gc.collect()
                # seed_everything(config.seed+10, workers=True)
                continue

            else:
                print('Model evaluation check passed!', flush=True)
                done = True

        return model

    def run_inference(self, model, dataloader, inference_type):
        """
        simple fn for running inference
        :param dataloader:      dataloader object for either labeled or unlabeled data
        :param model:           model object to perform inference with
        :param inference_type:  str to denote what kind of object inference should return (e.g. embeddings, predictions)
        :return:                numpy array of inference result (e.g. entropy for UNC, embeddings for DIV-F)
        """
        if inference_type not in ['max-entropy', 'embedding']:
            print('{} is an invalid inference type!'.format(inference_type), flush=True)
        model.acquisition_fn = inference_type
        trainer = get_trainer(self.config, logger=self.train_logger)
        print('Running inference..', flush=True)
        output = trainer.predict(model, dataloader)
        output = torch.cat(output, dim=0)
        output = output.numpy()

        return output

    def compute_input_diversity(self):
        """
        This function:
        1. loads previously selected indices and adds to labeled training set
        2. constructs sets of unique tokens for unlabeled and labeled training set
        3. computes jaccard similarity coefficient
        :return: None
        """

        self.init_dm()

        print('Computing input diversity', flush=True)
        indices = self.dm.train.load_past_indices(min_round=1,
                                                  relative_dataset=self.dm.train.U)
        self.dm.train.label_instances(indices, active_round=None, save_ids=False)

        def get_set_of_tokens(dataset):

            # join sentences from Premise and Hypothesis column into single string
            dataset['Combined'] = dataset['Premise'] + ' ' + dataset['Hypothesis']
            list_of_tokens = []
            # split each sentence into tokens and add to list
            for sentence in dataset['Combined'].tolist():
                for char in [',', '.', '?', '!', ':', ';',"'", '"']:
                    sentence = sentence.replace(char, '')
                list_of_tokens.extend(sentence.split())
            # return set of unique tokens
            return set(list_of_tokens)

        print('Constructing unique sets of tokens for unlabeled and labeled training sets', flush=True)
        unlabeled_set = get_set_of_tokens(dataset=self.dm.train.U)  # construct set of unique tokens in unlabeled set
        labeled_set = get_set_of_tokens(dataset=self.dm.train.L)  # construct set of unique tokens in labeled set

        intersection = unlabeled_set.intersection(labeled_set)  # get intersection between U and L
        union = unlabeled_set.union(labeled_set)  # get union of U and L

        print(len(union), flush=True)
        print(len(intersection), flush=True)

        input_diversity = len(intersection) / len(union)  # compute Jaccard similarity

        print('Input diversity: {}'.format(input_diversity), flush=True)
        self.log({'metric_input_diversity': input_diversity})

        return None

    def compute_feature_diversity(self):
        """
        This function computes the feature diversity of previously acquired data:
        1. train the model on the initial set of labeled data
        2. load the previously acquired data and move to 'labeled' set
        3. obtain embeddings for labelled and unlabelled set at test-time
        4. obtain distances for unlabelled features
        5. compute feature diversity
        6. log result to run-specific wandb file
        :return:
        """

        self.init_dm()

        print('Computing feature diversity', flush=True)
        # 1. Train model on initial data
        print('Training model on initial set of labeled data', flush=True)
        model = self._train_model()

        # 2. Load the 'acquired' data and put them in the 'labeled' pool, since U should be disjoint from L!
        print('\nLoading set of (previously acquired) data to add to labeled pool', flush=True)
        indices = self.dm.train.load_past_indices(min_round=1,
                                                  relative_dataset=self.dm.train.U)
        self.dm.train.label_instances(indices,
                                      active_round=None,
                                      save_ids=False)

        # 3. obtain embeddings for unlabelled and labelled data
        print('\nRunning inference on labelled data', flush=True)
        L_features = self.run_inference(model=model,
                                        dataloader=self.dm.labelled_dataloader(shuffle=False),
                                        inference_type='embedding')
        L_features = np.squeeze(L_features)

        print('\nRunning inference on unlabelled data', flush=True)
        U_features = self.run_inference(model=model,
                                        dataloader=self.dm.unlabelled_dataloader(shuffle=False),
                                        inference_type='embedding')
        U_features = np.squeeze(U_features)

        # 4. compute diversity score
        def get_per_element_score(acquired_features, unlabelled_features):
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(acquired_features)
            distances, indices = nbrs.kneighbors(unlabelled_features)
            return distances[:, 1]

        print('Finished inference.\nComputing distances between acquired batch and unlabelled set', flush=True)
        distances = get_per_element_score(acquired_features=L_features,
                                          unlabelled_features=U_features)

        feature_diversity = 1 / mean(distances)

        print('Feature diversity: {}'.format(feature_diversity), flush=True)
        self.log({'metric_feature_diversity': feature_diversity})

        return None
    
    def compute_uncertainty(self):

        self.init_dm()

        print('Computing uncertainty metric', flush=True)

        # TODO think of other settings that are exclusive to this kind of larger dataset run?

        # move everything to labelled set
        unlabelled_indices = self.dm.train.U.index.values.tolist()
        self.dm.train.label_instances(unlabelled_indices,
                                      active_round=None,
                                      save_ids=False)

        # train model on labelled set
        model = self._train_model()

        predictions = self.run_inference(model=model,
                                         dataloader=self.dm.labelled_dataloader(shuffle=False),
                                         inference_type='max-entropy')

        entropies = entropy(predictions, axis=1)


        # obtain indices of previously acquired data
        previous_indices = self.dm.train.load_past_indices(min_round=0,
                                                           relative_dataset=self.dm.train.L)

        metric_uncertainty = np.mean(np.array(entropies)[previous_indices])

        print('Uncertainty: {}'.format(metric_uncertainty), flush=True)
        self.log({'metric_uncertainty': metric_uncertainty})
