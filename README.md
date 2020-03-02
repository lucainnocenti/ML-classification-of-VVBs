Source code and data for the paper ***Machine learning-based classification of vector vortex beams***.

Authors: Taira Giordani, Alessia Suprano, Emanuele Polino, Francesca Acanfora, Luca Innocenti, Alessandro Ferraro, Mauro Paternostro, Nicoló Spagnolo, and Fabio Sciarrino.

***Add reference when paper is published.***

# Content

- [`notebooks`](./notebooks). Jupyer notebooks containing code and explanations to reproduce the results of the paper.
- [`src`](./src). The notebooks use a number of supporting functions, which are all defined here.
- [`data`](./data). Contains the datasets used in the notebooks.
- [`saved_CNN_models`](./saved_CNN_models). Pre-trained CNNs used in the notebooks.

# Requirements

The code uses Python 3. The necessary packages can be installed with either `conda` or `pip`.
To install all the required packages in one go, we include `requirements.txt` files.

If you use `conda`, you can create a new environment and replicate the one used to run these notebooks running the following command in a terminal (from the directory containing the `conda_requirements.txt` file):

```bash
conda create --name keras --file ./conda_requirements.txt
```

Alternatively, using `pip`, run the following:

```bash
pip install -r ./pip_requirements.txt
```
