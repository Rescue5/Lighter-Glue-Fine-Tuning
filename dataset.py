from pathlib import Path
from PIL import Image
import numpy as np
import torch
from modules.dataset.augmentation import AugmentationPipe
import os

class MatcherDataset(torch.utils.data.Dataset):
    """Some Information about MatcherDataset"""
    def __init__(self, device, img_dir: Path, img_size: tuple[int, int] =(1024, 1280), difficulty = 0.1, mode='train'):
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
            sides_crop=0,
            max_num_imgs=20,
            num_test_imgs=1,
            photometric=False,
            geometric=True,
        ).to(self.device)


    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

        res, ext = self.gomographPipe(img_tensor.to(self.device), self.difficulty, TPS=False)
        H, mask = ext  # H: [B,3,3], mask: [B,H,W]

        img0 = img_tensor.squeeze(0)      # [3,H,W]
        img1 = res.squeeze(0)             # [3,H,W]
        H    = H.squeeze(0)               # [3,3]
        mask = mask.squeeze(0)            # [H,W]

        return img0.cpu(), img1.cpu(), (H.cpu(), mask.cpu())

    def __len__(self):
        return len(self.img_paths)