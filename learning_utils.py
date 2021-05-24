import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class ImageDataset(torch.utils.data.Dataset):
    """ Used to create a DataLoader compliant Dataset for binary classifiers
    """
    def __init__(self, images,
                 standardize=True, mean=.5, std=.5,
                 to_grayscale=False, data_augmentation_transforms=None):
        """
        Args:
          data_augmentation_transforms (list): list of torchvision transformation for data augmentation.
          images (ndarray):
              The images.
              shape -> N x height x width x channels or N x height x width. dtype should be uint8.

        """
        #  Handle images ( self.images should be NxHxWxC even when C is 1, and image type should be uint8
        assert len(images.shape) == 3 or len(images.shape) == 4, \
            f'Expected images shape to be one of NxHxWxC or NxHxW. Shape given {images.shape}.'
        assert images.dtype == np.uint8, f'Images expected to have type uint8, not {images.dtype}.'
        assert 3 <= len(images.shape) <= 4, f'The images should be an array of shape N x H x W x [ C ] not {images.shape}'
        if len(images.shape) == 3:
            #  NxHxW  -> NxHxWx1
            images = images[..., None]
        self.n_images, self.height, self.width, self.n_channels = images.shape
        self.images = images

        self.mean = mean
        self.std = std

        # Handle transforms
        # ToPILImage Takes ndarray input with shape HxWxC
        transforms = [torchvision.transforms.ToPILImage()]

        if self.n_channels > 1 and to_grayscale:
            # Grayscale takes a PILImage as an input
            transforms.append(torchvision.transforms.Grayscale(num_output_channels=1))

        if data_augmentation_transforms is not None:
            transforms.extend(data_augmentation_transforms)

        # ToTensor accept uint8 [0, 255] numpy image of shape H x W x C and scales to [0, 1]
        transforms.append(torchvision.transforms.ToTensor())
        if standardize:
            # Standardization brings the mean of the dataset to 0 and the std of the dataset to 1 (centering and scaling)
            # if the mean and the std provided are the mean and std of the dataset.
            # If mean and std are .5 then, if the range of values of the images are 0, 1, then the values of the images
            # become -1 and 1.
            # Normalise takes input a tensor image of shape CxHxW and brings to target mean and standard deviation
            transforms.append(torchvision.transforms.Normalize((mean,), (std,)))

        self.transform = torchvision.transforms.Compose(transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()

        if self.transform and not torch.is_tensor(image):
            image = self.transform(image)

        return image


class LabeledImageDataset(ImageDataset):
    """ Used to create a DataLoader compliant Dataset for binary classifiers
    """
    def __init__(self, images, labels,
                 standardize=True, mean=.5, std=.5,
                 to_grayscale=False, data_augmentation_transforms=None):
        """
        Args:
          images (ndarray):
              The images.
              shape -> N x height x width x channels or N x height x width grayscale image.\
              Type must be uint8 ( values from 0 to 255)
          labels (ndarray):
              The corresponding list of labels.
              Should have the same length as images. (one label for each image)
        """
        super().__init__(images, standardize=standardize, mean=mean, std=std, to_grayscale=to_grayscale,
                         data_augmentation_transforms=data_augmentation_transforms)

        # if labels already ndarray nothing changes, if list makes to a numpy array
        labels = np.array(labels).squeeze()
        assert len(images) == len(labels), \
            f'Expected to have equal amount of labels and images. n images:{len(images)} n labels:{len(labels)}'
        assert labels.dtype in [np.int32, np.int64, np.int], f'Labels must be integers not {labels.dtype}'
        assert len(labels.shape) == 1, f'Labels should be a list of one label for each image, shape given {labels.shape}'

        self.labels = torch.from_numpy(labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def test_1():
    # A test to check that the LabeledImageDataset and ImageDataset yields the same results
    from generate_datasets import get_cell_and_no_cell_patches

    _, _, cell_images, non_cell_images, _, _, _ = get_cell_and_no_cell_patches()

    labeled_dataset = LabeledImageDataset(cell_images, np.ones((len(cell_images), ), dtype=np.int), standardize=True)
    image_dataset = ImageDataset(cell_images, standardize=True)

    assert len(labeled_dataset) == len(image_dataset), 'The number of samples in the two datasets should be the same.'

    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=len(labeled_dataset), shuffle=False)
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=len(image_dataset), shuffle=False)

    for images_1, (images_2, labels) in zip(image_loader, labeled_loader):
        assert images_1.allclose(images_2)

    labeled_dataset = LabeledImageDataset(cell_images, np.ones((len(cell_images), ), dtype=np.int), standardize=False)
    image_dataset = ImageDataset(cell_images, standardize=False)

    assert len(labeled_dataset) == len(image_dataset), 'The number of samples in the two datasets should be the same.'

    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=len(labeled_dataset), shuffle=False)
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=len(image_dataset), shuffle=False)

    for images_1, (images_2, labels) in zip(image_loader, labeled_loader):
        assert images_1.allclose(images_2)

    labeled_dataset = LabeledImageDataset(cell_images, np.ones((len(cell_images), ), dtype=np.int), standardize=True)
    image_dataset = ImageDataset(cell_images, standardize=False)

    assert len(labeled_dataset) == len(image_dataset), 'The number of samples in the two datasets should be the same.'

    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=len(labeled_dataset), shuffle=False)
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=len(image_dataset), shuffle=False)

    for images_1, (images_2, labels) in zip(image_loader, labeled_loader):
        assert not images_1.allclose(images_2)


def test_transformations():
    from generate_datasets import get_cell_and_no_cell_patches

    translation_pixels = 4
    final_patch_size = 31
    cell_image_creation_patchsize = final_patch_size + round(final_patch_size * .5) + translation_pixels
    if cell_image_creation_patchsize % 2 == 0:
        cell_image_creation_patchsize += 1

    translation_ratio = translation_pixels / cell_image_creation_patchsize

    _, _, cell_images, non_cell_images, _, _, _ = get_cell_and_no_cell_patches(
        try_load_from_cache=True,
        patch_size=cell_image_creation_patchsize
    )

    import PIL.Image
    labeled_dataset = LabeledImageDataset(
        cell_images[:2],
        np.ones((len(cell_images[:2]), ), dtype=np.int),
        standardize=False,
        data_augmentation_transformations=[
            torchvision.transforms.RandomAffine(degrees=(90, -90),
                                                translate=(translation_ratio, translation_ratio),
                                                resample=PIL.Image.BILINEAR,
                                                fillcolor=int(cell_images.mean())),
            torchvision.transforms.CenterCrop(final_patch_size)
        ])

    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=1, shuffle=False)
    from plotutils import cvimshow

    for _ in range(60):
        i = 0
        for images, labels in labeled_loader:
            if i % 2 == 0:
                print(i)
                print(images[0].squeeze().shape)
                cvimshow('window', images[0].squeeze().numpy())
            i += 1


if __name__ == '__main__':
    # test_1()
    test_transformations()
