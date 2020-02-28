import os
import sys
import collections
import glob
import tqdm
import PIL
import random
import imageio

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import skimage

from keras.utils import np_utils


def abs2(data):
    return np.abs(data)**2


def borderless_imshow_save(data, outputname, size=1, dpi=80, imshow_opts={}):
    width = 1 * size
    height = data.shape[0] / data.shape[1] * size
    size = (width, height)
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data, **imshow_opts)
    plt.savefig(outputname, dpi=dpi)
    plt.close(fig)


def imshow_intensities(amplitudes=None, intensities=None, imshow_opts={}, ax=None):
    """Plot imshow of squared modulus of amplitudes."""
    if amplitudes is None and intensities is None:
        raise ValueError('One of `amplitudes` and `intensities` must be given.')
    if amplitudes is not None and intensities is not None:
        raise ValueError('Only one of `amplitudes` and `intensities` must be given.')
    if amplitudes is not None:
        intensities = np.abs(amplitudes)**2
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.imshow(intensities, interpolation='nearest', cmap='magma',
              origin='lower', **imshow_opts)
    ax.axis('off')


def imshow_stokes_probs(prob_vectors, imshow_opts={}, axs=None, show_axis=True):
    """Plot stokes probabilities associated with a VVB state.
    This produces three plots, one per Stoke parameter.

    Attributes
    ----------
    prob_vectors : list of three probability vectors
        Should be of shape (3, N, M). If it is (N, M, 3) transpose is applied.
    """
    if prob_vectors.shape[2] == 3:
        prob_vectors = np.transpose(prob_vectors, (2, 0, 1))
    if axs is None:
        _, axs = plt.subplots(1, ncols=3, sharey=True)
    labels = ['0/1', '+/-', 'L/R']
    for ax, prob_vector, label in zip(axs, prob_vectors, labels):
        ax.imshow(prob_vector, **imshow_opts)
        ax.set_title(label)
        if not show_axis:
            ax.axis('off')
    return axs


def imshow_row_from_paths(paths, **kwargs):
    """Print a row of images from the list of given paths."""
    import imageio
    images = [imageio.imread(path) for path in paths]
    imshow_row(images, **kwargs)


def imshow_row(images, titles=None, plt_opts={}, axis='off',
               subplots_adjust=(0, 0.02), imshow_opts={}):
    """Print the given images in a row."""
    if titles is None:
        titles = ['' for _ in range(len(images))]
    fig, axs = plt.subplots(nrows=1, ncols=len(images), **plt_opts)
    if len(images) == 1:
        axs.imshow(images, **imshow_opts)
        axs.axis(axis)
        axs.set_title(titles[0])
    else:
        for ax, image, title in zip(axs, images, titles):
            ax.imshow(image, **imshow_opts)
            ax.axis(axis)
            ax.set_title(title)
    fig.subplots_adjust(hspace=subplots_adjust[0], wspace=subplots_adjust[1])


def plot_row(data, str_opts='', plt_opts={}):
    data = np.asarray(data)
    if data.ndim == 2 and data.shape[0] == 1:
        data = data[0]
    # with two dimensions, assume a list of data points (no xs, only ys)
    if data.ndim == 2:
        fig, axs = plt.subplots(nrows=1, ncols=data.shape[0], **plt_opts)
        for ax, ys in zip(axs, data):
            ax.plot(ys, str_opts)
        return fig, axs
    else:
        raise NotImplementedError('todo')


def rescale_array_values(array, range_, old_range=None):
    min_, max_ = range_
    if old_range is None:
        old_min = array.min()
        old_max = array.max()
    else:
        old_min, old_max = old_range
    return min_ + (array - old_min) / (old_max - old_min) * (max_ - min_)


def make_into_rgb_format(array):
    """Make an N x M x 3 array fit for the RGB format."""
    # rescale to that min and max are 0 and 255, respectively
    # NOTE: I am not so sure about the correctness of the rescaling, CHECK
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return rescale_array_values(array, [0, 255]).astype(np.uint8)


def plot_stokes_probs_as_rgb(stokes_probs, ax=None):
    """Plot VVB as a single RGB image."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.imshow(make_into_rgb_format(stokes_probs))
    ax.axis('off')
    return ax


def add_noise_to_array(data, noise_level=0.1):
    """Add white noise to the data."""
    data = np.asarray(data)
    range_ = data.max() - data.min()
    return data + np.random.randn(*data.shape) * (noise_level * range_)


def merge_dict_elements(dict_):
    """Merge all elements of the dictionary into a single array."""
    data = None
    for key in list(dict_):
        if data is None:
            data = dict_[key]
        else:
            data = np.vstack((data, dict_[key]))
    return data


def dict_of_arrays_to_labeled_array(dict_):
    """Convert dict of arrays into a single array plus an array of labels.

    This function assumes that the keys of the given dictionary are of the form
    `cXX`, where `c` is a single char, and `XX` some integer number.
    """
    data = None
    labels = None
    for key in dict_:
        new_data = dict_[key]
        new_labels = np.full(shape=dict_[key].shape[0], fill_value=key)
        if data is None:
            data = new_data
            labels = new_labels
        else:
            data = np.vstack((data, new_data))
            labels = np.append(labels, new_labels)
    return data, labels


def truncate_in_reduced_space(data, trained_pca, num_dimensions_left):
    """Use PCA to reduce the dimension, then truncate and go back.

    Attributes
    ----------
    data : np.array of shape num_samples x feature_size
        The data in the original dimension.
    trained_pca : sklearn.PCA instance
        Used to switch to the reduced space, where we truncate the vector,
        and then used again to switch back to the original dimension.
    num_dimensions_left : int
        All components in dimensions with index greater than this number are
        zeroed out (in the reduced space). Should not be larger than the number
        of dimensions that trained_pca reduces the data to.
    """
    reduced_data = trained_pca.transform(data)
    reduced_data[:, num_dimensions_left:] = 0
    return trained_pca.inverse_transform(reduced_data)


def compute_accuracies_per_label(true_labels, predicted_labels, labels_names=None):
    if labels_names is None:
        labels_names = list(set(true_labels))  # do not have to be numeric
    # convert the labels into integers to ease handling
    labels_indices = list(range(len(labels_names)))
    
    accuracies_per_label = collections.OrderedDict()
    for label_idx in labels_indices:
        # we want to compute the output accuracies for this specific label
        accuracies = np.zeros(shape=(len(labels_names),))
        # extract the elements corresponding to the currently considered true label
        true_labels_indices = np.where(true_labels == labels_names[label_idx])
        # true_labels_per_class = true_labels[true_labels_indices]
        predictions = predicted_labels[true_labels_indices]
        # we iterate over all the labels that the classifier associated with the
        # currently considered true label
        for predicted_label_name, count in collections.Counter(predictions).items():
            # extract the index associated with this predicted label
            predicted_label_idx = labels_names.index(predicted_label_name)
            # put the number of times this label was predicted in `accuracies`
            accuracies[predicted_label_idx] = count / len(predictions)
        accuracies_per_label[label_idx] = accuracies
    return accuracies_per_label


def cartesian_to_angles(data):
    norms = scipy.linalg.norm(data, axis=1)
    angles = np.zeros(shape=(norms.shape[0], data.shape[1] - 1))
    for angle_idx in range(data.shape[1] - 1):
        angles[:, angle_idx] = np.arccos(data[:, angle_idx + 1] / norms)
    data = np.concatenate((norms[:, None], angles), axis=1)
    return data


def to_spherical_coordinates(xyz):
    """Takes cartesian (x, y, z) and returns spherical (r, theta, phi)."""
    rho = xyz[0]**2 + xyz[1]**2
    r = np.sqrt(rho + xyz[2]**2)
    theta = np.arctan2(np.sqrt(rho), xyz[2])
    phi = np.arctan2(xyz[1], xyz[0])
    return r, theta, phi


def to_spherical_coordinates_vectorized(xyz):
    """Same as `to_spherical_coordinates`, but accepts vector inputs."""
    ptsnew = np.zeros_like(xyz)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew


def spherical_to_cartesian_coordinates(coords):
    """Given (r, theta, phi) returns the corresponding cartesian (x, y, z)."""
    r, theta, phi = coords
    z = r * np.cos(theta)
    rho = r * np.sin(theta)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z


def degrees_to_spherical_coords(theta, phi):
    """Takes angles measured in degrees, and returns the corresponding angles in radians.
    
    Given (theta, phi), returns (1, theta_r, phi_r),
    with theta_r, phi_r the angles in radians.
    """
    r = 1
    factor = np.pi / 180
    return r, factor * theta, factor * phi


def standardize_naming_class_folders(path):
    """Changes name of each file in the subdirectory of the given path.
    THIS IS POTENTIALLY DISASTROUS, BE CAREFUL!
    
    For each subdirectory names "X", it renames all the files it contains
    to follow a naming scheme of the form "X_000.png", "X_001.png" etc.
    """
    raise ValueError('This can be disastrous to run. Just copy-paste the code'
                     ' in your notebook and run it yourself if are sure.')
    dirs = glob.glob(os.path.join(path, '*'))
    for dir_ in dirs:
        files = glob.glob(os.path.join(dir_, '*'))
        for idx, file in enumerate(files):
            oldname = os.path.abspath(file)
            path, filename = os.path.split(oldname)
            classname = os.path.split(path)[1]
            base, ext = os.path.splitext(filename)
            newname = os.path.join(path, '{}_{:03}.png'.format(classname, idx))
            os.rename(oldname, newname)


def find_all_images_in_dir(path, images_ext='jpeg'):
    """Return all image files in subdirs of given path."""
    all_images_paths = sorted(glob.glob(os.path.join(path, '*/*.' + images_ext)))
    return np.asarray(all_images_paths)


def invert_dict(input_dict):
    """Assuming a bijection between keys and values, invert the mapping.
    
    Returns a dict of the form bi->ai from one of the form ai->bi.
    """
    values_ = list(input_dict.values())
    if len(values_) != len(set(values_)):
        raise ValueError('There are repeated values in the given dictionary.')
    return dict((v, k) for k, v in input_dict.items())


def serialize_class_directories(path):
    """Converts the list of subdirs into integer values.
    
    Returns a dict mapping names of dirs to integers.
    """
    from keras.preprocessing.image import ImageDataGenerator
    return ImageDataGenerator().flow_from_directory(path).class_indices


def load_images_and_labels_from_dir(path, images_ext='jpeg', image_size=(128, 128)):
    """Load all images from subdirectories of given path.
    
    Parameters
    ----------
    path : str
        Directory containing the dataset. This is expected to contain
        a number of subdirectories, one per class.
    images_ext : str
        All and only the files with this path will be considered.
    image_size : tuple
        Each loaded image is resized with this.
    
    Returns two elements: an array with all the images, and an array with
    all the labels corresponding to the images
    """
    # iterate through all the images and load them into memory
    all_images_paths = find_all_images_in_dir(path, images_ext)
    num_images = len(all_images_paths)
    all_images = np.zeros(shape=(num_images, *image_size, 3), dtype=np.float)
    labels = np.ones(num_images, dtype=np.int) * (-1)
    from keras.preprocessing.image import ImageDataGenerator
    class_indices = ImageDataGenerator().flow_from_directory(path).class_indices

    iterator = tqdm.tqdm(list(enumerate(all_images_paths)), position=0, leave=True)
    for idx, image_path in iterator:
        # which directory does the image belong to?
        image_dir = os.path.split(os.path.split(image_path)[0])[1]
        # extract integer of class corresponding to directory name
        labels[idx] = class_indices[image_dir]
        # load image
        all_images[idx] = np.asarray(PIL.Image.open(image_path).resize(image_size)) / 255
    return all_images, labels


def print_truth_table(true_labels, predicted_labels, classes_to_indices_dict=None):
    """Print truth table corresponding to given true and predicted labels.
    
    Parameters
    ----------
    true_labels : numpy 1d array
        List of (numeric) correct labels, corresponding to some dataset.
    predicted_labels : numpy 1d array
        List of (numeric) predicted labels. Each element of this should
        ideally be equal to the corresponding element of true_labels.
    classes_to_indices_dict : dict
        Dictionary mapping each class name to the corresponding numeric index.
        If not given, we use the index as label for each class.
    """
    if classes_to_indices_dict is None:
        classes_to_indices_dict = {k:k for k in set(true_labels)}
    # create sorted list of class names and corresponding indices
    list_of_labels_names = list(classes_to_indices_dict.keys())
    list_of_labels_indices = [classes_to_indices_dict[name]
                              for name in list_of_labels_names]
    # compute accuracy per class
    num_classes = len(list_of_labels_names)
    accuracies = []
    for label in list_of_labels_indices:
        correct_indices = np.argwhere(true_labels == label).flatten()
        counts = np.bincount(predicted_labels[correct_indices]).astype(np.float)
        counts = np.append(counts, np.zeros(num_classes - counts.shape[0]))
        counts /= correct_indices.shape[0]
        accuracies.append(counts)
    accuracies = np.array(accuracies)
    # display truth table
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.heatmap(accuracies, annot=True, cbar=False, square=True, ax=ax,
                xticklabels=list_of_labels_names, yticklabels=list_of_labels_names)


def filenames_from_list_of_paths(paths):
    """Returns the last bit of the paths given as input.
    
    E.g. from ['a/b', 'c/d'] we get ['b', 'd'].
    """
    return [os.path.split(path)[1] for path in paths]


def load_all_images_from_dict(dict_of_images, resize=(128, 128)):
    """Load all images into dict of classes.
    
    Each element of the input dict should contains a number of paths of images to load.
    The keys are the names of the classes.
    
    Returns
    -------
    A pair of arrays. The first one is the loaded dataset (as a single array), and
    the second one is the corresponding array of labels.
    """
    classes_names = sorted(list(dict_of_images.keys()))
    num_classes = len(classes_names)
    classes_to_idx_dict = {k:v for k,v in zip(classes_names, np.arange(num_classes))}
    total_num_imgs = sum(len(items) for items in dict_of_images.values())
    
    out_images = np.zeros(shape=(total_num_imgs, *resize, 3), dtype=np.float)
    out_labels = np.zeros(shape=(total_num_imgs, num_classes), dtype=np.float)
    labels_idx = 0
    imgs_idx = 0
    iterator = tqdm.tqdm(list(dict_of_images.items()), position=0, leave=True)
    for class_, images in iterator:
        # store labels
        out_labels[labels_idx:labels_idx + len(images)] = np_utils.to_categorical(
            y=classes_to_idx_dict[class_], num_classes=num_classes)
        labels_idx = labels_idx + len(images)
        # store images
        for image in images:
            out_images[imgs_idx] = skimage.transform.resize(
                image=imageio.imread(image) / 255,
                output_shape=resize
            )
            imgs_idx += 1
    return out_images, out_labels


def split_dict_into_two_dicts(dict_of_classes, dim_train_classes):
    """Split elements of dict into two disjoint dicts.
    
    Parameters
    ----------
    dict_of_classes : dict
        The dictionary we want to split
    dim_train_classes : int
        Number of elements in each class to put into the first output dict.
    """
    # randomly sample from each element of `dict_of_classes`
    train_dict = dict()
    test_dict = dict()
    for class_name, items in dict_of_classes.items():
        random.shuffle(items)
        train_dict[class_name] = items[:dim_train_classes]
        test_dict[class_name] = items[dim_train_classes:]
    return train_dict, test_dict


def split_dataset_into_train_and_test(original_path, dim_train_set, img_ext='jpeg'):
    """Take dataset in path and produce a train and test dataset from it.
    
    This loads all the images in the subdirs of the given path into memory, so
    be sure to have enough memory.
    
    Parameters
    ----------
    original_path : str
        Path containing class directories (and all of the data)
    dim_train_set : int
        The number of elements to include in each training class.
    img_ext : str
        Only files with this extension will be used.

    Returns
    -------
    A tuple with four elements: x_train, y_train, x_test, y_test
    """
    list_of_classes_dirs = glob.glob(os.path.join(original_path, '*'))
    num_images_per_class = [len(glob.glob(os.path.join(path, '*.' + img_ext)))
                            for path in list_of_classes_dirs]
    if any(dim_train_set > num for num in num_images_per_class):
        raise ValueError('Not enough images in the class directories.')
    # gather all images
    all_paths_dict = dict()
    for class_dir in list_of_classes_dirs:
        all_paths_dict[os.path.split(class_dir)[1]] = glob.glob(os.path.join(class_dir, '*.' + img_ext))
    # randomly sample `dim_train_set` images from each class
    train_paths_dict, test_paths_dict = split_dict_into_two_dicts(all_paths_dict, dim_train_set)
    # load all images into memory
    train_images = None
    x_train, y_train = load_all_images_from_dict(train_paths_dict)
    x_test, y_test = load_all_images_from_dict(test_paths_dict)
    return x_train, y_train, x_test, y_test


def save_data_into_files(x, y, classes_dict, new_path):
    """Take sets of images and labels and saves everything in a directory.
    
    The followed files are sorted into the usual directory structure.
    This is sort of an inverse operation of `split_dataset_into_train_and_test`.

    Parameters
    ----------
    x : numpy array
        Array of images.
    y : numpy array
        Each element of this is a label corresponding to an element of `x`.
        The labels are stored in categorical form, use np.argmax to convert them
        into integers.
    classes_dict : dict
        Dictionary mapping class labels into class indices.
    new_path : str
        Where to save the data.
    """
    all_labels = np.argmax(y, axis=1)
    print('Saving...', end='')
    for label, label_idx in classes_dict.items():
        os.makedirs(os.path.join(new_path, label))
        print(' {},'.format(label), end='')
        good_indices = np.argwhere(all_labels == label_idx).flatten()
        for idx, good_idx in enumerate(good_indices):
            imageio.imwrite(uri=os.path.join(new_path, label, '{}_{:03}.jpeg'.format(label, idx)),
                            im=(x[good_idx] * 255).astype(np.uint8))
    print()
    print('All done chief')