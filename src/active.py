# General imports
import argparse
import sys
import os
import wandb
import gc
import warnings
import logging
import datetime
import time

# Torch modules

# Local imports
from src.models import TransformerModel
from src.datasets import GenericDataModule
from src.strategies import select_acquisition_fn
from src.utils import log_results, log_percentages, get_trainer, get_model


def active_iteration(i, config, dm, logger):

    print('Active Learning iteration: {}\n'.format(i + 1), flush=True)

    # ---------------------------------- Training model on current labelled dataset ------------------------------
    # initialise model
    model = get_model(config)

    # initialise trainer
    trainer = get_trainer(config=config,
                          logger=logger)

    # train model
    model, dm, logger, trainer = train_model(dm=dm,
                                             config=config,
                                             model=model,
                                             logger=logger,
                                             trainer=trainer)

    # ---------------------------------------------  Evaluating model  -------------------------------------------
    # evaluate best checkpoint on dev set
    evaluate_model(split='dev',
                   dm=dm,
                   config=config,
                   model=model,
                   trainer=trainer,
                   logger=wandb)

    # ------------------------------------ Acquiring new instances for labeling ----------------------------------
    # Exit AL loop if all data was already labelled
    if dm.has_unlabelled_data() is False:
        break

    # initialise acquisition function
    acquisition_fn = select_acquisition_fn(fn_id=config.acquisition_fn, logger=logger)

    # determine instances for labeling using provided acquisition fn
    to_be_labelled = acquisition_fn.acquire_instances(config=config,
                                                      model=model,
                                                      dm=dm,
                                                      k=config.labelling_batch_size)

    # log share of each dataset in set of queried examples
    log_percentages(mode='active',
                    new_indices=to_be_labelled,
                    logger=wandb,
                    dm=dm,
                    epoch=i)

    # label new instances
    dm.train.label_instances(indices=to_be_labelled,
                             active_round=i + 1)

    # log composition of updated labelled pool
    log_percentages(mode='makeup',
                    new_indices=None,
                    logger=wandb,
                    dm=dm,
                    epoch=None)

    # some extra precautions to prevent memory leaks
    del model
    del trainer
    del acquisition_fn
    gc.collect()