from torch.utils.data import DataLoader
import pytorch_lightning.callbacks as cl
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from FER.datasets.custom1 import FERDataset
from FER.models import CustomCNNModule

def train():

    train_dts = FERDataset('data/FER2013Train', 'label.csv', (64, 64))
    val_dts = FERDataset('data/FER2013Valid', 'label.csv', (64, 64))

    loader_cfg = dict(batch_size=32, num_workers=32, drop_last=False)
    train_loader = DataLoader(train_dts, **loader_cfg, shuffle=True)
    val_loader = DataLoader(val_dts, **loader_cfg, shuffle=False)

    model = CustomCNNModule()
    
    logger = WandbLogger(project="FER")

    checkpoint_callback = cl.ModelCheckpoint(monitor="val_loss", dirpath="work_dir", 
        filename="Best_Checkpoint", save_top_k=1, mode="min")
    
    trainer_cfg = dict(
        accelerator='gpu',
        devices = 1,
        max_epochs=100,
        enable_progress_bar=True,
        default_root_dir='work_dir',
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=4,
        logger=logger,
        gradient_clip_val=1.,
    )
    trainer = pl.Trainer(**trainer_cfg)
    # trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=100, enable_progress_bar=True, default_root_dir='work_dir', check_val_every_n_epoch=2)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    train()
