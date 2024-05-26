import torch
from tqdm import trange
from utils.genDataset import synthetic_data
from utils.readDataset import data_iter
from utils.linRegFunc import linreg, squared_loss, sgd

# 1. Generate Dataset
w = torch.Tensor([2, -3.4])
b = 4.2
num_examples = 1000
features, labels = synthetic_data(w, b, num_examples)

# 2. Initialization
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 3. In each epoch
#   3a. Read Dataset (divide into batches)
#   3b. Calculate current loss
#   3c. Apply SGD, update parameters (suming up, minus its gradient)

batch_size = 10
num_epochs = 10
lr = 0.03
net = linreg
loss = squared_loss

for epoch in trange(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        
        sgd([w, b], lr, batch_size)

    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
