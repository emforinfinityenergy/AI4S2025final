import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os
import numpy as np
import pandas as pd


def masking(cr, img, x):
    mask = cv2.inRange(img, np.array(cr[0]), np.array(cr[1]))
    cl = mask[:, x]
    try:
        idx = np.where(cl > 0)[0][0]
    except IndexError:
        print(cr)
        return 10000000
    return idx


def smooth(idx, h):
    return max(1 - idx / h, 0)


def get_data(image_label, x):  #此函数用于使用cv2提取在浓度为x时的产物比例. This function is used to extract the product ratio at a concentration of x using cv2.
    # 读取图像，Read image
    img = cv2.imread(f"{image_label}")
    if img is None:
        raise ValueError(f"Cannot read the img: {image_label}.png")

    cropped_img = img[97:711,126:899]

    # 获取裁剪后图像尺寸，Get the size of the cropped image
    height, width = cropped_img.shape[:2]

    # 将x值映射到图像像素坐标，Map the x value to the image pixel coordinates.
    x_min, x_max = 2, 12  # 图像x轴范围，Image x-axis range.
    x_pixel = int((x - x_min) * width / (x_max - x_min))

    # 确保x_pixel在有效范围内，Ensure x_pixel is within the valid range.
    x_pixel = max(0, min(x_pixel, width-1))

    # 定义每种元素对应的颜色范围 (OpenCV读入自动默认BGR格式)，Define the color range corresponding to each element (OpenCV reads in BGR format by default).
    color_ranges = {
        'N2': ([160, 0, 0], [255, 140, 100]),       # 蓝色，Blue
        'NH4_ion': ([0, 110, 250], [120, 170, 255]), # 橘黄色，Orange
        'N2O': ([0, 160, 0], [150, 255, 150]),      # 绿色，Green
        'NO': ([0, 0, 160], [100, 100, 255]),        # 红色，Red
        'NO2': ([180, 100, 140], [200, 120, 160]),    # 紫色，Purple
    }
    #
    smoothed_results = {'N2': smooth(masking(color_ranges['N2'], cropped_img, x_pixel), height),
                        'NH4_ion': smooth(masking(color_ranges['NH4_ion'], cropped_img, x_pixel), height),
                        'N2O': smooth(masking(color_ranges['N2O'], cropped_img, x_pixel), height),
                        'NO': smooth(masking(color_ranges['NO'], cropped_img, x_pixel), height),
                        'NO2': smooth(masking(color_ranges['NO2'], cropped_img, x_pixel), height)}

    return (smoothed_results['N2'], smoothed_results['NH4_ion'], smoothed_results['N2O'],
            smoothed_results['NO'], smoothed_results['NO2'])


# datapath_train = "/bohr/train-gvtn/v1/"
datapath_train = ""
input_csv_path_train = os.path.join(datapath_train + 'input_train.csv')
ref_path = os.path.join(datapath_train + 'ref_result_train.csv')
data_train = pd.read_csv(input_csv_path_train)
ref_train = pd.read_csv(ref_path)


class EnvDataset(torch.utils.data.Dataset):
    def __init__(self, feature, ref):
        super().__init__()
        self.feature = feature
        self.ref = ref

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.ref[idx]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 8),
            nn.Tanh(),
            nn.Linear(8, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        return self.net(x)


features = []
refs = []
for i in range(5):
    features.append(get_data(data_train['File Name'][i], data_train['c'][i]))
    refs.append([ref_train['p_1'][i], ref_train['p_2'][i], ref_train['p_3'][i], ref_train['p_4'][i],
                 ref_train['p_5'][i], ref_train['p_6'][i], ref_train['p_7'][i], ref_train['p_8'][i]])

features = torch.tensor(features, dtype=torch.float32)
refs = torch.tensor(refs, dtype=torch.float32)

print(features)
print(refs)

dataset = EnvDataset(features, refs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)


NUM_EPOCHS = 40000

net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(NUM_EPOCHS):
    for feature, ref in dataloader:
        optimizer.zero_grad()
        pred = net(feature)
        loss = criterion(pred, ref)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        with torch.no_grad():
            pred = net(features)
            loss = criterion(pred, refs)
            print(f"epoch: {epoch}, loss: {loss}")


features = []
refs = []
i = 4
features.append(get_data(data_train['File Name'][i], data_train['c'][i]))
refs.append([ref_train['p_1'][i], ref_train['p_2'][i], ref_train['p_3'][i], ref_train['p_4'][i],
            ref_train['p_5'][i], ref_train['p_6'][i], ref_train['p_7'][i], ref_train['p_8'][i]])
pred = net(torch.tensor(features, dtype=torch.float32))
pred = pred.detach().numpy()
print(pred)
print(refs)
l = 0
for i in range(len(refs)):
    l += abs(refs[0][i] - pred[0][i])

print(l)
