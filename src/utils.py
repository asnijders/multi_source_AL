"""
This Python script will be used for any logic that does not belong to
a distinct component of the learning process
"""

from pytorch_lightning.utilities.memory import garbage_collection_cuda
import time
import gc
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging, Callback, LearningRateMonitor
from src.models import TransformerModel
import os
import sys

import pickle

class WriteTestPredictions(Callback):

    def on_test_epoch_end(self, trainer, pl_module):

        pl_module.reset_confidences()
        test_loader = trainer.test_dataloaders[0]

        print(test_loader)

        torch.set_grad_enabled(False)
        pl_module.eval()

        count = 0
        dataset_id = ''

        for batch in test_loader:

            count += len(batch['labels'])

            if dataset_id == '':
                dataset_id = batch['dataset_ids'][0]

            batch['input_ids'] = batch['input_ids'].to(pl_module.device)
            batch['token_type_ids'] = batch['token_type_ids'].to(pl_module.device)
            batch['attention_masks'] = batch['attention_masks'].to(pl_module.device)
            batch['labels'] = batch['labels'].to(pl_module.device)

            pl_module.datamap_step(batch)

        torch.set_grad_enabled(True)
        pl_module.write_confidences(type='test_predictions', dataset='/{}/'.format(dataset_id))

        return None

class DatamapCallback(Callback):

    def on_validation_epoch_end(self, trainer, pl_module):

        train_loader = trainer.train_dataloader

        torch.set_grad_enabled(False)
        pl_module.eval()

        print('Running inference for dataset cartography for epoch {}'.format(pl_module.current_epoch),
              flush=True)

        count = 0

        for batch in train_loader:

            count += len(batch['labels'])

            batch['input_ids'] = batch['input_ids'].to(pl_module.device)
            batch['token_type_ids'] = batch['token_type_ids'].to(pl_module.device)
            batch['attention_masks'] = batch['attention_masks'].to(pl_module.device)
            batch['labels'] = batch['labels'].to(pl_module.device)

            pl_module.datamap_step(batch)

        torch.set_grad_enabled(True)
        pl_module.train()

        return None

    def on_train_end(self, trainer, pl_module) -> None:

        pl_module.write_confidences()
        print('Finished dataset cartography.', flush=True)
        sys.exit()

        return None

class TestDatamapCallback(Callback):


    def on_validation_epoch_end(self, trainer, pl_module):

        test_dataloader = trainer.val_dataloaders[0]

        torch.set_grad_enabled(False)
        pl_module.eval()

        print('Running inference for dataset cartography on test set for epoch {}'.format(pl_module.current_epoch),
              flush=True)

        count = 0

        for batch in test_dataloader:

            count += len(batch['labels'])

            batch['input_ids'] = batch['input_ids'].to(pl_module.device)
            batch['token_type_ids'] = batch['token_type_ids'].to(pl_module.device)
            batch['attention_masks'] = batch['attention_masks'].to(pl_module.device)
            batch['labels'] = batch['labels'].to(pl_module.device)

            pl_module.datamap_step(batch)

        torch.set_grad_enabled(True)
        pl_module.train()

        return None

    def on_train_end(self, trainer, pl_module) -> None:

        pl_module.write_confidences()
        print('Finished dataset cartography for test set.', flush=True)
        sys.exit()

        return None


def create_project_filepath(config):
    """ Creates a logical filepath using provided experiment parameters """

    mode = ''
    if config.seed_size == 1.0:
        mode = 'full-supervision'
    else:
        mode = 'active-learning'

    filepath = 'project_{}/model_{}/data_{}/mode_{}'.format(config.project_dir,
                                                            config.model_id,
                                                            '-'.join(config.datasets),
                                                            mode)

    if config.checkpoint_datasets is not None:
        filepath += '/checkpoint-data_{}'.format('-'.join(config.checkpoint_datasets))

    return filepath


def log_results(logger, results, dm): # TODO put this in a Logger class

    # log training metrics
    results = results[0]
    for key in results.keys():
        res_dict = {'active_'+key: results[key], 'labelled_examples': len(dm.train.L)}
        logger.log(res_dict)

    return None


def set_val_check_interval(current_iter, total_iters):

    # initially the training set is small, and we don't want to evaluate every few examples.
    # so we set val_check_interval to 1.0 at AL_iter = 0. As acquisition progresses we evaluate more often
    # we evaluate 5 times per epoch at most (1/0.20 = 5)
    return max(0.20,
               (1.0 - (current_iter * 1/total_iters))
               )


def get_model(config, train_loader):
    """
    simple fn for initialising model
    """

    model = TransformerModel(model_id=config.model_id,
                             dropout=config.dropout,
                             lr=config.lr,
                             batch_size=config.batch_size,
                             acquisition_fn=config.acquisition_fn,
                             mc_iterations=config.mc_iterations,
                             num_gpus=config.gpus,
                             separate_test_sets=config.separate_test_sets,
                             train_loader=train_loader,
                             config=config)

    return model


def get_trainer(config, logger, batch_size=None, gpus=None):
    """
    simple fn for building trainer object
    :param gpus:
    :param config:
    :param logger:
    :param batch_size:
    :return:
    """

    if config.debug is False:

        mode = None
        if config.monitor == 'val_loss':
            mode = "min"
        elif config.monitor == 'val_acc':
            mode = "max"

        # Init early stopping
        early_stopping_callback = EarlyStopping(monitor=config.monitor,
                                                patience=config.patience,
                                                min_delta=0.01,
                                                verbose=True,
                                                mode=mode)

        # Init ModelCheckpoint callback, monitoring 'config.monitor'
        if config.train_difficulty is not None:
            train_difficulty = '/' + config.train_difficulty + '/'
        else:
            train_difficulty = '/'
        run_dir = config.checkpoint_dir + '/' + config.array_uid.replace(' ','_') + '/' + config.acquisition_fn + '/' + '_'.join(config.train_sets).replace('.','-') + train_difficulty  + str(config.seed) + '/'
        checkpoint_callback = ModelCheckpoint(monitor=config.monitor,
                                              mode=mode,
                                              save_top_k=1,
                                              dirpath=run_dir,
                                              filename='{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}',
                                              verbose=True)

        lr_monitor = LearningRateMonitor(logging_interval='step')

        callbacks = [early_stopping_callback,
                     checkpoint_callback,
                     lr_monitor]  # StochasticWeightAveraging(swa_lrs=1e-2)]

        if config.datamap:

            datamap_callback = DatamapCallback()
            callbacks.append(datamap_callback)

        if config.test_datamap:

            test_datamap_callback = TestDatamapCallback()
            callbacks.append(test_datamap_callback)

        if config.write_test_preds:

            test_pred_callback = WriteTestPredictions()
            callbacks.append(test_pred_callback)

        if config.training_acc_ceiling is not None:

            training_ceiling = EarlyStopping(monitor='train_acc_epoch',
                                            patience=999,
                                            stopping_threshold=config.training_acc_ceiling,
                                            verbose=True,
                                            mode="max",
                                            check_on_train_epoch_end=True)
            callbacks.append(training_ceiling)

        epochs = config.max_epochs
        val_check_interval = config.val_check_interval

    else:
        callbacks = None
        epochs = 20
        val_check_interval = 1

    if gpus is None:
        gpus = config.gpus

    trainer = Trainer(gpus=gpus,
                      strategy=config.strategy,
                      logger=logger,
                      callbacks=callbacks,
                      log_every_n_steps=config.log_every,
                      accelerator=config.accelerator,
                      max_epochs=epochs,
                      min_epochs=config.min_epochs,
                      deterministic=True,
                      enable_checkpointing=True,
                      enable_model_summary=True,
                      val_check_interval=val_check_interval,
                      num_sanity_val_steps=0,
                      limit_val_batches=config.toy_run,
                      limit_train_batches=config.toy_run,
                      limit_test_batches=config.toy_run,
                      progress_bar_refresh_rate=config.refresh_rate,
                      enable_progress_bar=config.progress_bar,
                      precision=config.precision,
                      overfit_batches=config.overfit_batches,
                      accumulate_grad_batches=config.accumulate_grad_batches,
                      track_grad_norm=2)

    return trainer


def evaluation_check(dm, config, model, trainer, current_results):

    print('WARNING: performing model check using aggregate dev sets', flush=True)
    checkpoint_loader = dm.val_dataloader()
    results = trainer.validate(model, checkpoint_loader)[0]

    print(results.keys(), flush=True)
    val_res = results['val_acc_epoch']

    print('Validation accuracy: {}'.format(val_res), flush=True)
    failure = None
    if val_res < config.model_check_threshold:

        print('Previous model failed. Re-training model for this iteration!', flush=True)
        failure = True
        return failure, val_res

    if len(current_results) > 0:

        # we restart training for this iteration if:
        #   current result is poorer than previous iter
        #   current result is only a marginal improvement w.r.t previous iter
        # if val_res - current_results[-1] < 0:
        #     failure = True

        # if the current result is more than 5 points lower than the previous result, restart training
        if val_res - current_results[-1] < -0.01:
            print('Difference between current iteration ({}) and last iteration ({}) is greater than 0.05. Re-training model for this iteration!'.format(val_res, current_results[-1]), flush=True)
            failure = True

        else:
            failure = False

    return failure, val_res


def train_model(dm, config, model, logger, trainer, inference_during_training=False):
    """
    Trains provided model on labelled data, whilst checkpointing on one or more datasets
    :param dm: Data Module obj
    :param config: argparse obj
    :param model: TransformerModel instance
    :param logger: logger obj
    :param trainer: trainer obj
    :return: saved model with lowest dev-loss
    """

    # initialise dataloaders for current data
    labelled_loader = dm.labelled_dataloader()  # dataloader with labelled training data

    # one can either checkpoint based on single, or multiple-dataset dev performance
    if config.checkpoint_datasets is None:
        print('\nCheckpointing model weights based on aggregate dev performance on: {}'.format(dm.val.L['Dataset'].unique()))
        checkpoint_loader = dm.val_dataloader()

    else:
        print('\nCheckpointing model weights based on dev performance on: {}'.format(config.checkpoint_datasets))
        # dev loader for specified checkpointing dataset(s)
        checkpoint_loader = dm.get_separate_loaders(split='dev',
                                                    dataset_ids=config.checkpoint_datasets)

    # fine-tune model on (updated) labelled dataset L, from scratch, while checkpointing model weights
    print('\nFitting model on updated labelled pool, from scratch', flush=True)

    trainer.fit(model=model,
                train_dataloaders=labelled_loader,
                val_dataloaders=checkpoint_loader)

    # print('CHECKPOINTING DISABLED!',flush=True)
    # return model checkpoint with lowest dev loss
    if config.debug is False:
        print('\nLoading checkpoint: {}'.format(trainer.checkpoint_callback.best_model_path))
        model = TransformerModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model, dm, logger, trainer


def evaluate_model(dm, config, model, trainer, logger, split):
    """
    fn for evaluating trained model on either:
    - separate test sets, in case of multiple datasets
    - single test set, in case of aggregate or single test set
    :param split:
    :param logger:
    :param trainer:
    :param dm: datamodule obj
    :param config: argparse obj
    :param model: trained transformer instance
    :return: dictionary with test statistics
    """

    if split == 'test':
        if config.separate_test_sets is True:
            print('Evaluating best model on separate {} sets'.format(split), flush=True)
            test_loaders = dm.get_separate_loaders(split=split,
                                                   dataset_ids=dm.test.L['Dataset'].unique())

            for test_loader, dataset_id in zip(test_loaders, dm.test.L['Dataset'].unique()):

                model.test_set_id = dataset_id + '_'
                model.init_metrics()
                print('Test results for {}'.format(dataset_id), flush=True)
                results = trainer.test(model, test_loader)
                # trainer.test_dataloaders = None

                # log test results
                log_results(logger=logger,
                            results=results,
                            dm=dm)

            model.test_set_id = ''

        else:
            print('Evaluating best model on aggregate {} set'.format(split), flush=True)
            test_loader = dm.test_dataloader()
            model.init_metrics()
            results = trainer.test(model, test_loader)

            # log test results
            log_results(logger=logger,
                        results=results,
                        dm=dm)

    elif split == 'dev':
        if config.separate_eval_sets is True:
            print('Evaluating best model on separate {} sets'.format(split), flush=True)
            dev_loaders = dm.get_separate_loaders(split=split,
                                                  dataset_ids=dm.val.L['Dataset'].unique())

            for dev_loader, dataset_id in zip(dev_loaders, dm.val.L['Dataset'].unique()):

                model.dev_set_id = dataset_id + '_'
                model.init_metrics()
                print('Validation results for {}'.format(dataset_id), flush=True)
                results = trainer.validate(model, dev_loader)

                # log test results
                log_results(logger=logger,
                            results=results,
                            dm=dm)

            # reset dev set identifier to empty string
            model.dev_set_id = ''

        else:
            print('Evaluating best model on aggregate {} set'.format(split), flush=True)
            dev_loader = dm.val_dataloader()
            model.init_metrics()
            results = trainer.validate(model, dev_loader)

            # log test results
            log_results(logger=logger,
                        results=results,
                        dm=dm)

        ### OOD evaluation
        if config.ood_sets is not None:

            print('Evaluating best model on OOD {} set'.format(split), flush=True)
            dev_loader = dm.separate_loader(sub_datapool=dm.ood_val)

            dataset_id = dm.ood_val.L['Dataset'].unique()[0]
            model.dev_set_id = dataset_id + '_'
            model.init_metrics()
            print('OOD Validation results for {}'.format(dataset_id), flush=True)
            results = trainer.validate(model, dev_loader)

            # log test results
            log_results(logger=logger,
                        results=results,
                        dm=dm)

            # reset dev set identifier to empty string
            model.dev_set_id = ''


def log_percentages(mode, new_indices, logger, dm, epoch):
    """
    Logs the makeup of the current labelled pool, or the AL iteration, in terms of:
    - dataset composition
    - label distribution
    - more other things in the future?
    :param epoch:
    :param mode: 'makeup' logs statistics for the current pool of labelled examples
                 'active' logs statistics for AL iteration
    :param new_indices: set of indices of newly queried examples
    :param logger: wandb object
    :param dm: datamodule
    :return:
    """

    for key in ['Dataset', 'Label']:

        # monitor composition of labelled pool over time
        if mode == 'makeup':

            labelled_examples = dm.train.L
            percentages = labelled_examples[key].value_counts(normalize=True).to_dict()
            percentages = {k + '_makeup': v for k, v in percentages.items()}
            percentages['labelled_examples'] = len(dm.train.L)  # variable for x-axis: current L
            logger.log(percentages)

        # monitor composition of each AL batch
        elif mode == 'active':

            new_examples = dm.train.U.iloc[new_indices]  # select queried examples from unlabelled pool
            percentages = new_examples[key].value_counts(normalize=True).to_dict()
            percentages['labelled_examples'] = len(dm.train.L) + len(new_examples)  # variable for x-axis: old L + new batch
            percentages['AL_iter'] = epoch
            logger.log(percentages)

            # determine the composition of the current pool of unlabelled examples
            unlabelled_pool_composition = dm.train.U['Dataset'].value_counts(normalize=True).to_dict()

            # determine the composition of the most recent batch of queries
            new_batch_composition = new_examples['Dataset'].value_counts(normalize=True).to_dict()

            # normalise composition of new batch by correcting for size of sub-dataset in unlabelled pool
            normalised_proportions = {k + '_normalized': new_batch_composition[k]/unlabelled_pool_composition[k] for k,_ in new_batch_composition.items()}
            normalised_proportions['AL_iter'] = epoch
            normalised_proportions['labelled_examples'] = len(dm.train.L) + len(new_examples)  # TODO add a counter for the already labeled examples
            logger.log(normalised_proportions)

    return None


def del_checkpoint(filepath, verbose=True):

    try:
        os.remove(filepath)
        if verbose:
            print('Removed checkpoint at {}!'.format(filepath), flush=True)

    except Exception:
        print('No checkpoint found for {}!'.format(filepath), flush=True)
        pass


def collect_garbage():

    garbage_collection_cuda()
    time.sleep(5)
    torch.cuda.empty_cache()
    garbage_collection_cuda()
    gc.collect()
