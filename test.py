import torch
from FER.models import CustomCnnModel
if __name__ == '__main__':
    model = CustomCnnModel(10)
    x = torch.ones((1, 1, 64, 64))
    y = model(x)
