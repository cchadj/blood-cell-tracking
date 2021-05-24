import torch
import matplotlib.pyplot as plt
import torchvision
from matplotlib.patches import Rectangle
import numpy as np


# thanks to: https://medium.com/@mh_yip/opencv-detect-whether-a-window-is-closed-or-close-by-press-x-button-ee51616f7088
def cvimshow(img, window='', wait_time=0):
    # imshow using cv2 with ability to close with the 'x' button without hanging in the main thread
    import cv2
    cv2.namedWindow(window, cv2.WINDOW_KEEPRATIO)
    if img.dtype == np.bool8:
        img = np.uint8(img) * 255
    cv2.imshow(window, img)
    cv2.waitKey()
    while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey()
        if (keyCode & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break


def savefig_tight(name, v=True):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    if v:
        print('Saved ', name)


def no_ticks(axes=None, hide_numbers=True):
    if axes is not None:
        axes = np.array(axes)
        ax: plt.axes
        for ax in axes.flatten():
            ax.tick_params(
                axis='both',
                which='both',
                left=False,
                right=False,
                labelleft=False,
                bottom=False,
                top=False,
                labelbottom=False)

            if hide_numbers:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
    else:
        plt.tick_params(
                axis='both',
                which='both',
                left=False,
                right=False,
                labelleft=False,
                bottom=False,
                top=False,
                labelbottom=False)
        ax = plt.gca()
        if hide_numbers:
            ax.set_yticklabels([])
            ax.set_xticklabels([])


def plot_patch_rois_at_positions(positions, patch_size=(21, 21), annotate=False,
                                 edgecolor='r', pointcolor='b', label=None,
                                 linestyle=None,
                                 ax=None, image=None, linewidth=2):
    assert type(patch_size) is int or type(patch_size) is tuple
    if type(patch_size) is int:
        patch_size = patch_size, patch_size
    patch_height, patch_width = patch_size

    if ax is None:
        _, ax = plt.subplots()
    if image is not None:
        ax.imshow(image, cmap='gray')

    positions = positions.astype(np.int32)
    ax.scatter(positions[:, 0], positions[:, 1], label=label, c=pointcolor)
    for patch_count, (x, y) in enumerate(positions.astype(np.int32)):
        rect = Rectangle((x - patch_width / 2,
                          y - patch_height / 2),
                         patch_width, patch_height, linestyle=linestyle,
                         linewidth=linewidth, edgecolor=edgecolor, facecolor='none')

        ax.add_patch(rect)
        if annotate:
            ax.annotate(patch_count, (x, y))


def plot_dataset_as_grid(dataset, title=None):
    """ Plots a stack of images in a grid.

    Arguments:
        dataset: The dataset
        title: Plot title
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=60000,
        shuffle=False
    )

    for batch in loader:
        images = batch[0]
        labels = batch[1]

        print("Images:", images.shape)
        print("Labels:", labels.shape)
        grid_img = torchvision.utils.make_grid(images, nrow=50)

        plt.figure(num=None, figsize=(70, 50), dpi=80, facecolor='w', edgecolor='k')
        plt.title(title)
        plt.grid(b=None)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()


def plot_images_as_grid(images, ax=None, title=None, figsize=(20, 5), fontsize=30):
    """
    Plots a stack of images in a grid.

    Arguments:
        images: The images as NxHxWxC
        title: Plot title
    """
    if len(images.shape) == 3:
        images = images[..., None]

    batch_tensor = torch.from_numpy(images)
    # NxHxWxC -> NxCxHxW
    batch_tensor = batch_tensor.permute(0, -1, 1, 2)
    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=90)
    if ax is None:
        _, ax = plt.subplots(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    plt.grid(b=None)
    ax.imshow(grid_img.permute(1, 2, 0))
    ax.tick_params(
        axis='both',
        which='both',
        left=False,
        right=False,
        labelleft=False,
        bottom=False,
        top=False,
        labelbottom=False)