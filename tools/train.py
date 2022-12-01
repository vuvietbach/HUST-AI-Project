from torch.utils.data import DataLoader
import pytorch_lightning.callbacks as cl
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from FER.datasets import FERDataset
from FER.models import CustomCNNModule

def train():
    batch_size = 32

    train_dts = FERDataset('csv_file/train.csv')
    val_dts = FERDataset('csv_file/val.csv', mode='test')

    train_loader = DataLoader(train_dts, batch_size=batch_size, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dts, batch_size=batch_size, num_workers=8, shuffle=False)

    model = CustomCNNModule()
    
    #logger = WandbLogger(project="FER")

    # checkpoint_callback = cl.ModelCheckpoint(monitor="validation_loss", dirpath="work_dir", filename="Best_Checkpoint",
    #     save_top_k=1, mode="min")
    # trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=100, enable_progress_bar=True, default_root_dir='work_dir', callbacks=[checkpoint_callback]\
    #     , logger=logger, check_val_every_n_epoch=4)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10, enable_progress_bar=True, default_root_dir='work_dir', check_val_every_n_epoch=4)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    train()
