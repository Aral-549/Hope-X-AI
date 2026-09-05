"""
Data Augmentation Pipelines for Natural Scene Classification.

Domain Justification for Natural Scenes:
1. Horizontal Flip: Natural landscapes (mountains, forests, glaciers, sea) and streetscapes
   remain semantically and physically valid when mirrored horizontally.
2. Random Affine / Shear & Scale (0.8 - 1.2): Simulates varied camera viewpoints, distances,
   and slight angular perspective variations common in user-submitted landscape photos.
3. Mild Color Jitter (brightness/contrast): Captures real-world diurnal changes, cloud cover,
   and weather variance across outdoor scenes without distorting vegetation or sea tones.
"""

import torchvision.transforms as transforms


def get_baseline_transforms(image_size: int = 150):
    """Clean deterministic baseline transforms without stochastic augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def get_advanced_transforms(image_size: int = 150):
    """
    Carefully curated augmentations designed specifically for 6-class natural scenes.
    Prevents overfitting on small sample sizes (600 to 3000 images).
    """
    train_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, shear=5, scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def get_test_transforms(image_size: int = 150):
    """Standard evaluation and test inference transform."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
