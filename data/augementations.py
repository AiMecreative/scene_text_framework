import torch
import torch.nn as nn
import kornia.augmentation as K


class ImageAugmentation(K.AugmentationBase2D):

    def __init__(self, color: bool, geometry: bool):
        super().__init__()

        self.color = color
        self.geometry = geometry

        self.color_pipeline = nn.Sequential(
            K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
            K.RandomGaussianNoise(std=0.2, p=0.15),
            K.RandomGaussianBlur(3, (3, 3), p=0.25),
        )
        self.geometry_pipeline = nn.Sequential(
            K.RandomAffine((-10.0, 10.0), (0, 0.1), (0.8, 1.2), (5, 15), p=0.5),
            K.RandomPerspective(0.15, p=0.25),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        By default, this method is used to handle batched images,
        but here we use in dataset __getitem__ only
        """
        if self.color:
            images = self.color_pipeline(images)
        if self.geometry:
            images = self.geometry_pipeline(images)
        return images
