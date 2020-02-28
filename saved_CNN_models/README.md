# CNN training attempts log

## Train&test on './simulated 15 classes'

This contains training and test dataset for VVBs belonging to 15 classes, classified by their (m1, m2) OAM quantum numbers.

NOTE: Some experimental images are, for some reason, inside this folder. The trained CNNs are found to only misclassify these images. In other words, all the following nets achieve 100% accuracy on the simulated images.

- '1convlayer.h5'. A single conv+maxpool layer. Maxpooling with (2, 2) window. Uses ~16M pars.

- '1convlayer_5x5pools.h5'. As '1convlayer.h5', but now we maxpool with a 5x5 window. This has ~2.5M pars.

- '4convlayers_5x5firstmaxpool.h5'. Has ~35k pars.

- '2convs_10and5pools_32dense.h5'. 5k pars. Still good enough.

Training only with simulated images, even introducing very strong shearing, zooming, shifting in the training data, doesn't get us very far in managing to classify experimental images. Simulated images just don't seem to be a good enough model of the experimental ones.

## Train&test on './experimental/15classes'

We get good accuracies by splitting the dataset and using 20 random images per class for the training. The splitting is done with

````
utils.split_dataset_into_train_and_test(
    original_path='./data/experimental/15classes',
    dim_train_set=20
)
````

***Saved models***

- '2convs_5x5pools_32dense.h5'. Two conv layers with 16 3x3 filters each. Max-pooling with 5x5 window. Single hidden dense layer with 32 neurons. Trained on 20 exp imgs per class, achieves 0.92 acc on training set and 0.96 on test (the rest of the exp imgs).

- '2convs_5x5pools_32dense_exp2expGoodAcc.h5'. With 30x200 steps gives 0.98 accuracy on test.

- '3convs_432pools_64dense_exp2expGreatAcc.h5'. 38k parameters. With 100x200 steps it learnt to classify experimental images with 0.99 accuracy. The training was done with 20 exp images per class, using heavy preprocessing to distort and move images at the training stage, trying to only leave the feature we are interested in as useful predictor.