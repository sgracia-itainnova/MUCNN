import torch.utils.data as data
import torch
import numpy as np
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path)

        self.ms = dataset.get("ms")
        self.lms = dataset.get("lms")
        self.pan = dataset.get("pan")
        # print(self.ms.shape)
        # print(self.pan.shape)
        # print(dataset)
        # input()

    #####必要函数
    def __getitem__(self, index):
        return torch.from_numpy(self.lms[index, :, :, :]/ 2047).float(), \
               torch.from_numpy(self.ms[index, :, :, :]/ 2047).float(), \
               torch.from_numpy(self.pan[index, :, :, :]/ 2047).float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]

    def plot_ms_ground_truth(self, index, file_name):
        import matplotlib.pyplot as plt

        for i in range(self.ms.shape[1]):
            plt.figure()
            plt.imshow(self.ms[index, i, :, :], cmap='gray')
            plt.title(f'Channel {index + 1}')
            plt.axis('off')
            plt.savefig(f'{file_name}_channel_{index + 1}.png')

    def plot_pan_ground_truth(self, index, file_name):
        import matplotlib.pyplot as plt

        for i in range(self.pan.shape[1]):
            plt.figure()
            plt.imshow(self.pan[index, i, :, :], cmap='gray')
            plt.title(f'Channel {index + 1}')
            plt.axis('off')
            plt.savefig(f'{file_name}_channel_{index + 1}.png')
