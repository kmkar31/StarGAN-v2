from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import os
import random
from PIL import Image
import torch

# Each Pytorch dataset must have an __init__ method, a __getitem__ method and a __len__ method


class CelebA(Dataset):
    # Hyperparameters are included in the config file
    # root_dir is the absolute location to the dataset directory
    def __init__(self, root_dir, config, transform=False):
        self.root_dir = root_dir
        self.config = config
        self._init_dataset()
        if transform:
            self._init_transform()

    def _init_dataset(self):
        # Each "domain" is listed as a separate folder in the dataset
        domains = os.listdir(self.root_dir)
        fnames_1, fnames_2, labels = [], [], []

        for indx, domain in enumerate(sorted(domains)):
            # Pull all file names in each domain sorted alphabetically
            img_files = sorted(
                glob(os.path.join(self.root_dir, domain, '*.jpg')))
            fnames_1 += img_files
            # Get a random permutation of the file names in each domain
            fnames_2 += random.sample(img_files, len(img_files))
            # Assign label numbers to the domains based on alphabetical order
            labels += [indx]*len(img_files)
        # After going through each domain:
        # fnames_1 has all image file names sorted alphabetically domain-wise
        # fnames_2 has a shuffled list of file names still ordered domain-wise

        self.src_imgs = fnames_1
        # Create tuples of an image and another random image in the SAME DOMAIN for each domain
        self.ref_imgs = list(zip(fnames_1, fnames_2))
        self.src_labels = labels
        self.ref_labels = labels
        self._shuffle()

    def _shuffle(self):
        # Create tuples of SORTED source images and corresponding labels
        temp = list(zip(self.src_imgs, self.src_labels))
        # Shuffle the list of tuples
        random.shuffle(temp)
        # Reassign src_imgs and src_labels to contain the images in random order
        self.src_imgs, self.src_labels = zip(*temp)
        self.src_imgs = list(self.src_imgs)
        self.src_labels = list(self.src_labels)

        # Similarly shuffle the coupled images to be passed into the generator
        temp = list(zip(self.ref_imgs, self.ref_labels))
        random.shuffle(temp)
        self.ref_imgs, self.ref_labels = zip(*temp)
        self.ref_imgs = list(self.ref_imgs)
        self.ref_labels = list(self.ref_labels)

    def _init_transform(self):
        # A crop of random size (default: 0.08 to 1.0) of the original size and
        # a random aspect ratio (default: of 9/10 to 11/10) of the original aspect ratio is made.
        # This crop is finally resized to given size.
        crop = transforms.RandomResizedCrop(self.config["img_size"], scale=[
                                            0.8, 1.0], ratio=[0.9, 1.1])
        # Crop Image only for a certain number of images
        rand_crop = transforms.Lambda(lambda x: crop(
            x) if random.random() < self.config["prob"] else x)
        self.transform = transforms.Compose([
            # Randomly Crop an Image
            rand_crop,
            # Definitely resize the image irrespective of cropping
            transforms.Resize(
                [self.config["img_size"], self.config["img_size"]]),
            # Flip the image horizontally with a probablity of 0.5 (default)
            transforms.RandomHorizontalFlip(),
            # Convert the image to a tensor
            transforms.ToTensor(),
            # Normalize the pixel values of each channel to lie in a gaussian distribution with a mean of 0.5 and a standard deviation of 0.5
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        # Grt the file name of the image corresponding to that particular index
        src = self.src_imgs[index]
        ref1, ref2 = self.ref_imgs[index]
        src_label = self.src_labels[index]
        ref_label = self.ref_labels[index]
        # Open the file as an RGB image
        src = Image.open(src).convert('RGB')
        ref1 = Image.open(ref1).convert('RGB')
        ref2 = Image.open(ref2).convert('RGB')
        # Apply a transformation optionally
        if self.transform is not None:
            src = self.transform(src)
            ref1 = self.transform(ref1)
            ref2 = self.transform(ref2)
        # Convert the label from an integer to a torch tensor
        src_label = torch.tensor(src_label, dtype=torch.long)
        ref_label = torch.tensor(ref_label, dtype=torch.long)

        # Get two latent tensors of specified length sampled from a normal diistribution
        latent1 = torch.randn(self.config["latent_dim"])
        latent2 = torch.randn(self.config["latent_dim"])

        return src, src_label, ref1, ref2, ref_label, latent1, latent2

    def __len__(self):
        # Return the number of images in the Dataset
        return len(self.src_imgs)
