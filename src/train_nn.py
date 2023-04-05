import datetime
import os
from functools import partial

import wandb
from pytorch_lightning import seed_everything
from torchmetrics import MetricCollection, Accuracy, F1Score

from config import cfg
from utils.data_pipeline import convert_to_full_paths
from utils.experimentation import partition_dataset, perform_sweep_iter

GLOBAL_SEED = cfg['GLOBAL_SEED']


def perform_experiment():
    """
    Perform a model training experiment.
    """
    seed_everything(cfg['GLOBAL_SEED'])

    wandb_entity = cfg['WANDB']['ENTITY']
    wandb_project = 'SYDE544_FINALPROJECT_TESTING'

    # DREAMER
    dm_cfg = cfg['DATASETS']['DREAMER']
    dm_processed_data_path = dm_cfg['PROCESSED_DATA_PATH']
    dm_subjects = dm_cfg['SUBJECTS']
    dm_subjects = convert_to_full_paths(dm_subjects, dm_processed_data_path)

    data_sources = [dm_subjects]

    # DATALOADER ARGS #
    batch_size = cfg['BATCH_SIZE']
    shuffle = cfg['SHUFFLE']
    num_workers = cfg['NUM_WORKERS']

    data_loader_args = {'batch_size': batch_size,
                        'shuffle': shuffle,
                        'num_workers': num_workers}

    # MODEL AND SAVE PATH #
    # Saving paths (model, splits, checkpoints)
    model_def = cfg['MODEL_ARCHITECTURE']

    save_dir = os.path.join(cfg['SAVE_MODEL_PATH'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # NUM EPOCHS AND FOLDS #
    num_epochs = cfg['EPOCHS']
    num_folds = cfg['CV_FOLDS']

    # TRAINER ARGS #
    classes = cfg['CLASSES']
    trainer_args = {'classes': classes}

    # Metrics
    num_classes = len(classes)
    metrics = MetricCollection({
        'Accuracy': Accuracy(task="multiclass", num_classes=num_classes, average='macro'),
        'Multiclass F1-Score': F1Score(task='multiclass', num_classes=num_classes, average=None),
        'Macro F1-Score': F1Score(task='multiclass', num_classes=num_classes, average='macro'),
    })
    trainer_args['metrics'] = metrics

    # CALLBACK ARGS #
    callback_args = cfg['CALLBACKS']
    callback_args['MODEL_CHECKPOINT']['dirpath'] = save_dir

    test_set_percentage = cfg['TEST_SET_PERCENTAGE']
    train_set, test_set = partition_dataset(data_sources, test_percentage=test_set_percentage, seed=GLOBAL_SEED,
                                            save_dir=save_dir)

    sweep_cfg = cfg['HYPERPARAMETER_SEARCH']['SWEEP_SETTINGS']
    sweep_cfg['parameters'] = cfg['HYPERPARAMETER_SEARCH'][model_def]

    sweep_id = wandb.sweep(sweep_cfg, project=wandb_project, entity=wandb_entity)

    # Perform hyperparameter sweep and collect best scores
    wandb.agent(sweep_id, function=partial(perform_sweep_iter,
                                           train_set=train_set,
                                           test_set=test_set,
                                           model_def=model_def,
                                           trainer_args=trainer_args,
                                           callback_args=callback_args,
                                           data_loader_args=data_loader_args,
                                           num_epochs=num_epochs,
                                           num_folds=num_folds,
                                           save_dir=save_dir), count=cfg['HYPERPARAMETER_SEARCH']['N_EVALS'])

    # Stop sweep
    wandb.api.stop_sweep(sweep=sweep_id, entity=wandb_entity, project=wandb_project)


if __name__ == "__main__":
    perform_experiment()
    print('Done.')
