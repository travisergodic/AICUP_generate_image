import os
import logging

from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


image_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

class FidDataset(Dataset):
    def __init__(self, image_dir, type, transform=None):
        super().__init__()
        assert type in ("road", "river")
        self.image_dir = image_dir
        self.type = type
        self.transform = transform
        self.paths = sorted([
            os.path.join(image_dir, file) for file in os.listdir(image_dir) \
                if (file.split("_")[1] == {"road": "RO", "river": "RI"}[type]) and file.endswith(".jpg")
        ])
        logger.info(f"Found {len(self.paths)} of {type} paths")

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        image = self.transform(image)
        return {"data": image, "name": os.path.basename(self.paths[index])}
    
    def __len__(self):
        return len(self.paths)
    

def build_fid_loader(image_dir, type, batch_size):
    image_ds = FidDataset(image_dir, type=type, transform=image_transform)
    return DataLoader(image_ds, batch_size=batch_size, shuffle=False,  num_workers=2)