import os

import h5py
import mat73
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

mean = (
    15.9144,
    15.9394,
    12.1088,
    27.6384,
    26.1122,
    21.0799,
    14.1105,
    13.8744,
    13.8895,
)

std = (
    9.8100,
    10.2362,
    8.0946,
    11.2562,
    12.9910,
    10.1495,
    8.0082,
    7.4262,
    9.5949,
)

transform = transforms.Compose([
    transforms.Normalize(mean, std)  # Chuẩn hóa dữ liệu
])
class WiPoseDataset(Dataset):
    def __init__(self, root_dir, split="Train"):
        self.root_dir = root_dir
        self.split = split
        self.file_list = os.listdir(os.path.join(root_dir, split))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.split, self.file_list[idx])
        data_mat = mat73.loadmat(file_path)
        csi = data_mat["CSI"]
        csi_amp = np.array(csi).transpose(3, 2, 1, 0).reshape((9, 30, 5))
        CSI_data = torch.tensor(csi_amp)
        CSI_data = transform(CSI_data)
        keypoints = torch.tensor(
            np.array(data_mat["SkeletonPoints"]).reshape((3, 18)).T
        )
        xy_keypoints = keypoints[:, :2] * 0.001
        confidence_score = keypoints[:, 2:3]
        keypoints = torch.cat([xy_keypoints, confidence_score], dim=1)
        return {"input_wifi-csi": CSI_data, "output": keypoints}


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset = WiPoseDataset("/home/jackson-devworks/Desktop/HPE/Wi-Pose")
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    # Tính mean và std
    mean = torch.zeros(9)
    std = torch.zeros(9)
    for data in tqdm(loader):
        images = data["input_wifi-csi"]
        mean += images.mean(dim=[0, 2, 3])
        std += images.std(dim=[0, 2, 3])
    mean /= len(loader)
    std /= len(loader)

    print(f"Mean: {mean}")
    print(f"Std: {std}")
