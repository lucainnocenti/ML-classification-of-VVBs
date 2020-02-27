# Guide to dataset folder

## Some things that have been renamed at some point or the other

- `simulated 15 classes/test_set_balls_red` -> `simulated 15 classes/test`
- `simulated 15 classes/train_set_balls_red` -> `simulated 15 classes/train`

## Description of different datasets

- `simulated15classes`. Simulated VVBs generated as 300x300 images, divided into 15 classes as per the paper (no train/test division).
- `simulated15classes_noise0.1`. Same as `simulated15classes` but with normal noise added on the final intensities.
- `simulated 15 classes`. This is the dataset generated and used by Taira. It's similar to the aboves ones but the images have a number of pixels equal to the experimental images. This dataset is also divided into train and test.

# Used for CNN training and classification
These datasets have been used in the `CNNs for VVBs.ipynb` notebook to train/test CNNs.

- `./simulated 15 classes`. Contains dataset of *simulated* VVBs separated into 15 classes. Each class corresponds to random VVBs with OAM numbers (m1, m2) for some m1, m2. It contains two subdirectories, one used for training and one used for testing the CNNs. Each of these contains 15 directories, one per class.

- `./experimental/15classes`. Contains experimental VVBs corresponding to the same 15 classes of `./simulated 15 classes`.