from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


CALLBACKS = {
    'EARLY_STOPPING': EarlyStopping,
    'MODEL_CHECKPOINT': ModelCheckpoint
}
