import torch

def customCrossEntropyLoss(input, target):
    eps = 1e-4
    x = -target * torch.log(input+eps)
    return torch.mean(x)

def calculate_accuracy(input, target):
    pred = torch.argmax(input, dim=1)
    res = torch.mean(torch.tensor(pred==target, dtype=torch.float32))
    return res

def read_data():
    pass

def normalize_binary(frame, smooth=1e-3):
    return 