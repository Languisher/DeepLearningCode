import torch
from tqdm import trange

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
        
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in trange(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

        train_loss, train_acc = train_metrics
        print(f"Epoch {epoch + 1}: Train acc: {train_acc}, Test acc: {test_acc}")

def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
        
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        # We divide cases of optimizer being torch built-in func or user-defined func
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])

        metric.add(float(l.sum()),
                   accuracy(y_hat, y),
                   y.numel())
           
    return metric[0] / metric[2], metric[1] / metric[2]

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
       
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())

    return metric[0] / metric[1]

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
