"""
Dataset definitions, transforms mapping for 3LC tables, and test evaluation loader.
Rule: Test data is flat and strictly never registered or used in training tables.
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TestDataset(Dataset):
    """
    Flat test folder loader: all 1,800 test images without ground truth labels.
    Returns (image_tensor, image_id).
    """
    def __init__(self, image_dir: Path, transform=None, image_size: int = 150):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_size = image_size
        self.images = []

        if self.image_dir.exists():
            seen = set()
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img in self.image_dir.glob(ext):
                    key = img.name.lower()
                    if key not in seen:
                        seen.add(key)
                        self.images.append(img)
        self.images.sort(key=lambda x: x.name)
        print(f"[DATA] Loaded {len(self.images)} test images from {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load image {img_path}: {e}")
            image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        image_id = img_path.stem
        return image, image_id


class TableSampleMapper:
    """Wrapper to map 3LC table sample rows into (tensor, label) pairs for DataLoader."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image = Image.open(sample["image"])
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), sample["label"]
