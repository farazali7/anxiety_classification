import torch
from pytorch_lightning.trainer import Trainer
from classification.src.models.models import get_model, load_model_from_checkpoint


ckpt = 'results/models/20230331-135550/epoch=0--val_Macro F1-Score=0.00--fold=1.ckpt'

torch_load = torch.load(ckpt)

model = load_model_from_checkpoint(ckpt)

trainer = Trainer()

