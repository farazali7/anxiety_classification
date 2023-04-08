import os

cfg = {
    'DATASETS': {
        'DREAMER': {
            'RAW_DATA_PATH': 'data/raw/DREAMER',
            'FORMATTED_DATA_PATH': 'data/formatted/iter1/DREAMER',
            'PROCESSED_DATA_PATH': 'data/processed/iter2/DREAMER',
            'SUBJECTS': [str(i) for i in range(1, 24)],
            'SAMPLING_FREQ': 128
        },
        'DASPS': {
            'RAW_DATA_PATH': 'data/raw/dasps/open_hand',
            'FORMATTED_DATA_PATH': 'data/formatted/iter1/dasps',
            'PROCESSED_DATA_PATH': 'data/processed/iter1/dasps',
            'SUBJECTS': [str(i) for i in range(1, 24)],
            'SAMPLING_FREQ': 128
        },
    },
    'SAVE_MODEL_PATH': 'results/models',
    'SAVE_SPLITS_PATH': 'results/models',

    # Preprocessing args
    'BUTTERWORTH_ORDER': 2,
    'BUTTERWORTH_FREQ': [4, 45],

    # Feature extraction args
    'STANDARDIZE': True,  # Set to False to normalize data into [-1, 1] range instead (MaxAbsScaler)
    'FEATURE_EXTRACTION_FUNC': 'feature_set_1',

    'MODEL_ARCHITECTURE': 'CNN',
    'EXPERIMENT_TYPE': 'train',

    # Training & validation args
    'BATCH_SIZE': 128,
    'EPOCHS': 40,
    'LR': 0.001,
    'SHUFFLE': True,
    'NUM_WORKERS': os.cpu_count(),

    # Data partitioning args
    'CV_FOLDS': 4,
    'TEST_SET_PERCENTAGE': 0,

    'COMBINE_CHANNELS': True,

    'CLASSES': ['LVLA', 'LVHA', 'HVLA', 'HVHA'],

    'BATCH_SPECIFIC_TRAIN': False,  # If train batches should be specific to 1 subject at a time (ex. for AdaBN scheme)

    'GLOBAL_SEED': 7,

    'CALLBACKS': {
        # 'EARLY_STOPPING': {
        #     'monitor': 'val_Macro F1-Score',
        #     'min_delta': 0.001,
        #     'patience': 8
        # },
        'MODEL_CHECKPOINT': {
            'filename': '{epoch}--{val_Macro F1-Score:.2f}',
            'monitor': 'val_Macro F1-Score',
            'mode': 'max',
            'auto_insert_metric_name': True
        }
    },

    'HYPERPARAMETER_SEARCH': {
        'SWEEP_SETTINGS': {
            "method": "bayes",
            "metric": {
                "name": "Mean CV F1-Score",
                "goal": "maximize"
            },
        },
        'N_EVALS': 3,
        'MLP': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.5
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        },
        'MLP_ITER2': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.5
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        },
        'CNN': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 0.7
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        },
        'CNN_ITER2': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.5
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        },
        'CNN_ITER3': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.5
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        },
        'CNN_ITER4': {
            "dropout": {
                "distribution": "uniform",
                "min": 0.3,
                "max": 0.5
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01
            }
        }
    },

    'WANDB': {
        'PROJECT': 'SYDE544_FINALPROJECT',
        'ENTITY': 'fali'
    },

    # Fine-tuning parameters such as whether to run script locally (adjust paths), how many reps to use for training
    'FINETUNE': {
        'RUN_LOCALLY': False,
        'TRIALS': 3,
        'TEST_SET_SUBJECTS_PATH': 'results/models/20230401-200055/test_set.pkl',
        'CHECKPOINT_PATH': 'results/models/20230401-200055/epoch=21--val_Macro F1-Score=0.00--fold=1-v1.ckpt',  # Pre-trained model
        'EPOCHS': 40,
        'BATCH_SIZE': 32,
        'CALLBACKS': {
            # 'EARLY_STOPPING': {
            #     'monitor': 'val_Macro F1-Score',
            #     'min_delta': 0.001,
            #     'patience': 8
            # },
            'MODEL_CHECKPOINT': {
                'filename': '{epoch}--{train_loss:.2f}',
                'monitor': 'train_loss',
                'mode': 'min',
                'auto_insert_metric_name': True
            }
        },
        'REDUCE_LR': True,
        'PERFORM_MAJORITY_VOTING': True,
        'VOTERS': 3
    },
}
