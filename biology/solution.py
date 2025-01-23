import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dna, e_score):
        super().__init__()
        self.dna = dna
        self.e_score = e_score

    def __len__(self):
        return len(self.dna)

    def __getitem__(self, idx):
        return self.dna[idx], self.e_score[idx]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)



t1 = time.time()
df = pd.read_csv("train.csv")
dna = []
e_scores = []

for i in range(df.shape[0]):
    seg = df['DNA'][i]
    e_scores.append(df['E-score'][i])
    idx_l = ['', 'A', 'C', 'G', 'T']
    seg_l = []
    for n in seg:
        for j in range(4):
            seg_l.append(0)
        seg_l[-1 * idx_l.index(n)] = 1
    dna.append(seg_l)


print(dna)

dna_t = torch.tensor(dna, dtype=torch.float32)
e_scores_t = torch.tensor(e_scores, dtype=torch.float32)

dataset = MyDataset(dna_t, e_scores_t)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

NUM_EPOCHS = 5000
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    for dna_seg, e_score_seg in dataloader:
        optimizer.zero_grad()
        pred = net(dna_seg)
        loss = criterion(pred.T[0], e_score_seg)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        with torch.no_grad():
            pred = net(dna_t)
            loss = criterion(pred.T[0], e_scores_t)
            print(f"Epoch {epoch}, loss = {loss}, Time Elapsed: {time.time() - t1}")


# GGGTATCA
print(net(torch.tensor([3, 3, 3, 4, 1, 4, 2, 1], dtype=torch.float32)))
# CCGGCCGG
print(net(torch.tensor([2, 2, 3, 3, 2, 2, 3, 3], dtype=torch.float32)))
print(time.time() - t1)
