import random
from PIL import Image
from diffusion_ffpe.my_utils import make_dataset, build_transform
import torch.utils.data as data
import torchvision.transforms.functional as F


class UnpairedDataset(data.Dataset):
    def __init__(self, source_folder, target_folder, image_prep=None):
        super().__init__()
        self.source_folder = source_folder
        self.target_folder = target_folder

        self.l_imgs_src = make_dataset(self.source_folder, shuffle=True, seed=0)
        self.l_imgs_tgt = make_dataset(self.target_folder, shuffle=True, seed=0)

        self.T = build_transform(image_prep)

    def __len__(self):
        return len(self.l_imgs_src)

    def __getitem__(self, index):
        img_path_src = self.l_imgs_src[index]
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])

        return {"pixel_values_src": img_t_src, "pixel_values_tgt": img_t_tgt}
