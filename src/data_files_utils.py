import collections
import glob
import itertools
import logging
import time
import math
import os
import sys
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.special
import skimage
import sklearn
import sklearn.decomposition

import progressbar
from tqdm import tqdm

from . import utils
from .utils import abs2
from . import vector_vortex_beams as VVB


def generate_VVB_superpositions_dataset(
    pars_to_generate, data_dir, num_samples_per_class=50, noise_level=0.1,
    polarization_state='sequential phases', monitor=False,
    X=None, Y=None, w0=None, output_shape=(50, 50)
):
    if X is None or Y is None or w0 is None:
        raise ValueError('These are not really mandatory arguments')
    dataset = VVB.VVBDataset(X=X, Y=Y, w0=w0)
    dataset.generate_data(parameters=pars_to_generate,
                          num_samples=num_samples_per_class,
                          noise_level=noise_level,
                          polarization_state=polarization_state, monitor=monitor)
    dataset.labels = np.asarray(dataset.labels)
    # create root folder for new data (raise error if already existing)
    os.makedirs(data_dir)
    # min and max are used for the rescaling later
    minmax = [dataset.dataset.min(), dataset.dataset.max()]
    # iterate over all classes of generated images
    for label_idx, labels_name in enumerate(dataset.labels_names):
        m1, m2 = labels_name[1]
        # generate folder for new class
        class_name = 'm{:+d}{:+d}'.format(m1, m2)
        class_path = os.path.join(data_dir, class_name)
        os.makedirs(class_path)
        # all image arrays corresponding to a given class
        imgs = dataset.dataset[dataset.labels == label_idx]
        for img_idx, img in enumerate(imgs):
            img_filename = class_name + '({:03}).png'.format(img_idx)
            # reduce number of pixels and rescale image
            img = np.transpose(img.reshape((3, *X.shape)), [1, 2, 0])
            img = skimage.transform.resize(img, output_shape=output_shape, mode='reflect')
            img = utils.rescale_array_values(img, [0, 1], old_range=minmax)
            plt.imsave(os.path.join(class_path, img_filename), img)


class ImageDataFolder:
    def __init__(self, dir, img_formats=['png', 'jpeg']):
        if not os.path.isdir(dir):
            raise ValueError('Not a directory.')
        self.class_dirs = glob.glob(os.path.join(dir, '*/'))
        if len(self.class_dirs) == 0:
            raise ValueError('No subfolders found in the given path.')
        image_files = []
        for class_dir in self.class_dirs:
            files_in_dir = []
            for ext in img_formats:
                files_in_dir += glob.glob(os.path.join(class_dir, '*.' + ext))
            image_files += files_in_dir
        if len(image_files) == 0:
            raise ValueError('No images found in the given path.')
        self.image_files = image_files
    
    def split_into_train_and_test_dirs(self, new_root_path, train_ratio=0.5):
        num_new_images = int(len(self.image_files) * train_ratio)
        # decide which images to use for train and which for testing
        train_images = np.random.choice(self.image_files,
                                        size=num_new_images, replace=False)
        test_images = [img for img in self.image_files if img not in train_images]
        # create new directories in the given path
        new_train_path = os.path.join(new_root_path, 'train')
        new_test_path = os.path.join(new_root_path, 'test')
        os.makedirs(new_train_path)
        os.makedirs(new_test_path)
        # copy the images where due
        def copyfiles(infiles, outpath):
            for image_path in infiles:
                dir_ = os.path.dirname(image_path)
                class_name = os.path.split(dir_)[-1]
                class_dir = os.path.join(outpath, class_name)
                if not os.path.isdir(class_dir):
                    os.makedirs(class_dir)
                outimgpath = os.path.join(class_dir, os.path.split(image_path)[-1])
                shutil.copyfile(src=image_path, dst=outimgpath)
        copyfiles(infiles=train_images, outpath=new_train_path)
        copyfiles(infiles=test_images, outpath=new_test_path)


class ImagesDatasetToReduce:
    def __init__(self, data_dir, images_shape=None, pca=None,
                 n_components=None, monitor=None):
        self.data_dir = data_dir
        # -- determine dimension of each image
        if images_shape is None:
            if pca is None:
                raise ValueError('If the PCA is not trained I need explicit '
                                 'specs for the shapes of the images.')
            # we assume that each image has NxNx3 pixels
            pixel_width = int(np.sqrt(pca.components_[0].shape[0] // 3))
            self.images_shape = (pixel_width, pixel_width)
        else:
            self.images_shape = images_shape
        # -- determine with pca reducer we want to use
        if pca is None:
            if n_components is None:
                raise ValueError('If the PCA is not trained I need to know to '
                                 'how many dimensions you want to reduce the '
                                 'data.')
            # in this case we also train the pca on the dataset
            print('Training PCA on given dataset'); time.sleep(.3)
            self.pca = train_pca_on_files_in_dir(
                self.data_dir, n_components, output_shape=self.images_shape,
                monitor=monitor)
        else:
            self.pca = pca

        # -- load the images and store reduced representations

        # produce a dictionary with the keys being the labels and the
        # associated data the data arrays corresponding to the label
        print('Computing reduced representation of the images'); time.sleep(.3)
        self.reduced_images, self.filenames = load_and_reduce_images_in_dir(
            data_dir=self.data_dir, reducer=self.pca,
            output_shape=self.images_shape,
            return_classes=True, monitor=monitor,
            return_filenames=True
        )
        # convert into two arrays, the first with the full dataset as an array
        # and the second one with the labels associated with each element
        # of the first
        self.merged_data, self.labels = utils.dict_of_arrays_to_labeled_array(
            self.reduced_images)
    
    def train_svc(self, kernel, gamma='auto'):
        self.svc = sklearn.svm.SVC(kernel=kernel, gamma=gamma)
        self.svc.fit(self.merged_data, self.labels)

    def test_svc_accuracy(self, svc):
        predictions = svc.predict(self.merged_data)
        return collections.Counter(predictions == self.labels)


def load_resize_and_flatten_rgb_image(filename, output_shape):
    return skimage.transform.resize(
        plt.imread(filename)[:, :, :3],
        output_shape=output_shape, mode='reflect'
    ).flatten()


def load_images_in_dir(data_dir, output_shape=(50, 50), monitor=False,
                       return_classes=False):
    data_dir_handler = ImageDataFolder(data_dir)
    images_paths = data_dir_handler.image_files  # all images in dir
    images_size = 3 * output_shape[0] * output_shape[1]  # length of each img
    iterator = list(enumerate(images_paths))
    if monitor:
        iterator = progressbar.progressbar(iterator)

    images = np.empty(shape=(len(images_paths), images_size))
    if not return_classes:
        for idx, img_path in iterator:
            images[idx] = load_resize_and_flatten_rgb_image(
                filename=img_path, output_shape=output_shape)
    else:
        raise NotImplementedError('Not implemented yet')
    return images


def train_pca_on_files_in_dir(data_dir, n_components, output_shape=(50, 50),
                              monitor=False):
    ipca = sklearn.decomposition.IncrementalPCA(n_components=n_components)
    dirs = glob.glob(os.path.join(data_dir, '*/'))
    iterator = list(enumerate(dirs))
    if monitor:
        iterator = progressbar.progressbar(iterator)
    for _, dir_ in iterator:
        filenames = glob.glob(os.path.join(dir_, '*.jpeg'))
        filenames += glob.glob(os.path.join(dir_, '*.png'))
        filenames = np.asarray(filenames)
        # load images and do the pca
        images = np.zeros(shape=(len(filenames), output_shape[0] * output_shape[1] * 3))
        for idx, filename in enumerate(filenames):
            images[idx] = load_resize_and_flatten_rgb_image(
                filename,
                output_shape=output_shape
            )
        ipca.partial_fit(images)
    return ipca


def load_and_reduce_images(image_paths, reducer, output_shape=(50, 50)):
    """Load and apply PCA to a list of image files."""
    images = np.zeros(shape=(len(image_paths),
                             output_shape[0] * output_shape[1] * 3))
    for idx, filename in enumerate(image_paths):
        images[idx] = load_resize_and_flatten_rgb_image(
            filename, output_shape=output_shape)
    return reducer.transform(images)


def load_and_reduce_images_in_dir(data_dir, reducer, output_shape=(50, 50),
                                  monitor=False, return_classes=False,
                                  return_filenames=False):
    """Load and apply PCA to all images in subfolders of the given directory.

    Attributes
    ----------
    return_classes : bool
        If False, the images are returned all in a single array.
        If True, the output is a dict with the (reduced) images stored in each
        key element.
    """
    dirs = glob.glob(os.path.join(data_dir, '*/'))

    if return_filenames:
        all_filenames = []
    if return_classes:
        outdata = collections.OrderedDict()
    else:
        outdata = None
    iterator = list(enumerate(dirs))
    if monitor:
        iterator = progressbar.progressbar(iterator)
    for _, dir_ in iterator:
        filenames = glob.glob(os.path.join(dir_, '*.jpeg'))
        filenames += glob.glob(os.path.join(dir_, '*.png'))
        if return_filenames:
            all_filenames += filenames
        # load images and do the pca
        images = np.zeros(shape=(len(filenames),
                                 output_shape[0] * output_shape[1] * 3))
        for idx, filename in enumerate(filenames):
            images[idx] = load_resize_and_flatten_rgb_image(
                filename, output_shape=output_shape)
        reduced_images = reducer.transform(images)
        if return_classes:
            dirname = os.path.split(os.path.dirname(dir_))[-1]
            outdata[dirname] = reduced_images
        else:
            if outdata is None:
                outdata = reduced_images
            else:
                outdata = np.concatenate(
                    (outdata, reduced_images), axis=0)
                # outdata = np.vstack((outdata, reduced_images))
    if return_filenames:
        return outdata, all_filenames
    else:
        return outdata


def generate_labels_array_from_dir(data_dir):
    dirs = glob.glob(os.path.join(data_dir, '*'))
    
    out_labels = None
    for dir_idx, dir_ in enumerate(dirs):
        filenames = glob.glob(os.path.join(dir_, '*.jpeg'))
        filenames += glob.glob(os.path.join(dir_, '*.png'))
        # load images and do the pca
        labels = np.ones(shape=len(filenames), dtype=np.int) * dir_idx
        if out_labels is None:
            out_labels = labels
        else:
            out_labels = np.concatenate((out_labels, labels), axis=0)
    return out_labels
