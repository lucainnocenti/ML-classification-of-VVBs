Source code and data for the paper:

> Taira Giordani, Alessia Suprano, Emanuele Polino, Francesca Acanfora, Luca Innocenti, Alessandro Ferraro, Mauro Paternostro, Nicoló Spagnolo, and Fabio Sciarrino, ***Machine learning-based classification of vector vortex beams***, [Phys. Rev. Lett. 124, 160401, (2020)](https://doi.org/10.1103/PhysRevLett.124.160401), [arXiv:2005.07949](https://arxiv.org/abs/2005.07949).


# Content

- [`notebooks`](./notebooks). Jupyer notebooks containing code and explanations to reproduce the results of the paper.
- [`src`](./src). The notebooks use a number of supporting functions, which are all defined here.
- [`data`](./data). Contains the datasets used in the notebooks.
- [`saved_CNN_models`](./saved_CNN_models). Pre-trained CNNs used in the notebooks.

# Requirements

The code uses Python 3. The necessary packages can be installed with either `conda` or `pip`.
To install all the required packages in one go, we include `environment.yml` and `requirements.txt` files.

If you use `conda`, you can create a new environment and replicate the one used to run these notebooks running the following command in a terminal (from the directory containing `environment.yml`):

```bash
conda env create -f environment.yml
```

Alternatively, using `pip`, run the following:

```bash
pip install -r requirements.txt
```

Note that some of the notebooks assume the same directory structure of the repo is mantained. Running notebooks saved in different locations will require to manually make sure that the code in `./src` is available to be imported in the notebook.
