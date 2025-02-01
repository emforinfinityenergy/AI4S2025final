import math

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Parameter(torch.tensor(1., dtype=torch.float64, requires_grad=True), requires_grad=True)
        self.l2 = nn.Parameter(torch.tensor(1., dtype=torch.float64, requires_grad=True), requires_grad=True)

    def forward(self, x):
        return 2 * (1 - x) * self.l2 + 2 * x * self.l1


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, c):
        self.x = x
        self.c = c

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.c[idx]


NUM_EPOCHS = 10000


def cal(df_path):
    df = pd.read_csv(df_path)
    x_t = torch.tensor(df['x'], dtype=torch.float64)
    c_t = torch.tensor(df['C'], dtype=torch.float64)

    dataset = Dataset(x_t, c_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        for x, c in dataloader:
            optimizer.zero_grad()
            pred = net(x)
            loss = criterion(pred, c)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                pred = net(x_t)
                loss = criterion(pred, c_t)
                print(f'epoch {epoch}, loss: {loss}')

    a = net.l2.item()
    l = net.l1.item()
    print(a, l)
    return math.sqrt(3) * (a ** 2) / 12 * math.sqrt(l ** 2 - a ** 2 / 3)


if __name__ == '__main__':
    print(cal("C_train.csv"))
