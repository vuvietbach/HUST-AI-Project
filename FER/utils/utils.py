import torch

def customCrossEntropyLoss(input, target):
    t = torch.eye(7).to(input.device)
    t1 = t[target]
    t2 = t1 * input
    t3 = torch.sum(t2, dim=1)
    t4 = -1 * torch.log(t3)
    t5 = torch.mean(t4)
    return t5

def calculate_accuracy(input, target):
    pred = torch.argmax(input, dim=1)
    res = torch.mean(torch.tensor(pred==target, dtype=torch.float32))
    return res