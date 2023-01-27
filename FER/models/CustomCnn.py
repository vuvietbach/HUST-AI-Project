import torch.nn as nn
import torch
import pytorch_lightning as pl
from FER.utils import customCrossEntropyLoss
class CustomCNNBlock(nn.Module):
    def __init__(self, in_chan, out_chan, conv_kernel, maxpool_kernel=(2, 2)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, conv_kernel, padding=int(conv_kernel[0]/2)),
            nn.ReLU(),
            nn.BatchNorm2d(out_chan),
            nn.MaxPool2d(maxpool_kernel),
        )
    def forward(self, X):
        return self.features(X)

class CustomDenseBlock(nn.Module):
    def __init__(self, in_chan, out_chan, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_chan, out_chan),
            nn.ReLU(),
            nn.BatchNorm1d(out_chan),
            nn.Dropout(dropout),
        )
    def forward(self, X):
        return self.features(X)

class CustomCnnModel(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        num_chan = 8
        self.num_class = num_class
        self.features = nn.Sequential(
            CustomCNNBlock(1, num_chan, (3, 3)),
            CustomCNNBlock(num_chan, num_chan*2, (3, 3)),
            CustomCNNBlock(num_chan*2, num_chan*4, (3, 3)),
            CustomCNNBlock(num_chan*4, num_chan*8, (3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            CustomDenseBlock(1024, 512),
            CustomDenseBlock(512, 128),
            CustomDenseBlock(128, 32),
            nn.Linear(32, num_class),            
            nn.Softmax(dim=1)
        )
        self._initialize_weights()
    
    def forward(self, X):
        X1 = self.features(X)
        X2 = self.classifier(X1)
        return X2
    
    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



class CustomCNNModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_func = customCrossEntropyLoss
        self.model = CustomCnnModel(10)

    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95**epoch)
        return {'optimizer':optimizer, 'scheduler':scheduler}
    
    def training_step(self, batch, batch_idx):
        input, label_gt = batch
        
        label_pred = self.model(input)

        loss = self.loss_func(label_pred, label_gt)
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input, label_gt = batch
        
        label_pred = self.model(input)

        loss = self.loss_func(label_pred, label_gt)
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

if __name__ == '__main__':
    model = CustomCnnModel(10)
    X = torch.ones((1, 1, 64, 64))
    y = model(X)
    print(y.shape)