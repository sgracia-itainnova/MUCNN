import torch
import cv2
import numpy as np
from model.Model import Unet
from lib.data import DatasetFromHdf5
import scipy.io as sio
import os
import matplotlib.pyplot as plt


def get_edge(data):  # get high-frequency
    rs = np.zeros_like(data)
    if len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
    else:
        rs = data - cv2.boxFilter(data, -1, (5, 5))
    return rs


def load_set(file_path):
    data = DatasetFromHdf5(file_path)  # HxWxC
    index = 0 # Get the first element of the dataset
    data.plot_ms_ground_truth(index, file_name=f"ms_ground_truth")
    data.plot_pan_ground_truth(index, file_name=f"pan_ground_truth")
    # tensor type:
    lms = torch.from_numpy(data.lms[index] / 2047)        # CxHxW = 8x256x256
    ms_hp = torch.from_numpy((data.ms[index] / 2047))     # CxHxW= 8x64x64
    pan_hp = torch.from_numpy((data.pan[index] / 2047)).squeeze(dim=0)  # HxW = 256x256
    return lms, ms_hp, pan_hp

# ==============  Main test  ================== #


ckpt = "/home/sgracia/terravision/pansharpening/MUCNN/Weights/600.pth"   # chose model


def test(file_path, filename):
    lms, ms_hp, pan_hp = load_set(file_path)

    model = Unet(1, 8).cuda().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)

    with torch.no_grad():

        x1, x2, x3 = lms, ms_hp, pan_hp   # read data: CxHxW (numpy type)
        print(x1.shape)
        x1 = x1.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        x2 = x2.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        x3 = x3.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW

        aux1, aux2, hp_sr = model(x2, x3)  # tensor type: CxHxW
        # sr = x1 + hp_sr        # tensor type: CxHxW
        sr = hp_sr        # tensor type: CxHxW

        sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy()

        print(sr.shape)

        save_name = "output.mat"
        # sio.savemat(save_name, {'output_mucnn': sr})
        # Visualize individual channels
        for i in range(sr.shape[2]):
            plt.figure()
            plt.imshow(sr[:, :, i], cmap='gray')
            plt.title(f'Channel {i + 1}')
            plt.axis('off')
            plt.savefig(f'{save_name}_channel_{i + 1}.png')


if __name__ == '__main__':
    file_path = "/home/sgracia/terravision/pansharpening/MUCNN/data"
    for dir in os.listdir(file_path):
        test(os.path.join(file_path, dir), dir.split('.')[0])
