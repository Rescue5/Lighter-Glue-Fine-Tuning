from pathlib import Path
from PIL import Image
import numpy as np
import torch
from modules.dataset.augmentation import AugmentationPipe
import os

class MatcherDataset(torch.utils.data.Dataset):
    """Some Information about MatcherDataset"""
    def __init__(self, device, img_dir: Path, img_size: tuple[int, int] =(1280, 1024), difficulty = 0.1, mode='train'):
        super(MatcherDataset, self).__init__()
        self.img_dir = Path(os.path.join(img_dir,mode))
        self.img_paths = sorted(str(img) for img in self.img_dir.iterdir() if img.suffix.lower() in ['.jpg','.png'])
        self.img_size = img_size
        self.H = img_size[0]
        self.W = img_size[1]
        self.device = device
        self.difficulty = difficulty

        
        self.gomographPipe = AugmentationPipe(
            device=self.device,
            img_dir=str(self.img_dir),
            warp_resolution=self.img_size,
            out_resolution=self.img_size,
            sides_crop=0.2,
            max_num_imgs=20,
            num_test_imgs=1,
            photometric=True,
            geometric=True,
        ).to(self.device)


    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0)

        res, ext = self.gomographPipe(img_tensor.to(self.device), self.difficulty, TPS=False)
        return img_tensor.squeeze(), res.squeeze().permute(0,2,1), ext

    def __len__(self):
        return len(self.img_paths)