# Datasets

- `./simulated15classes`. Simulated VVBs generated as 300x300 images, divided into 15 classes as per the paper (no train/test division). It contains 15 subdirectories, one per class. The class are defined by pairs of OAM numbers (m1, m2).
- `./simulated15classes_noise0.1`. Same as `simulated15classes` but with normal noise added on the final intensities.
- `./simulated 15 classes`. Similar to the aboves ones but the images have here the same size as the experimental ones. Contains two subdirs, one for training and one for testing, each one of which has the same structure as `simulated15classes`.
- `./experimental/15classes`. Contains experimental VVBs corresponding to the same 15 classes of `./simulated 15 classes`.