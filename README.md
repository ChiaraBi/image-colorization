# Image colorization
Project for the university course of Vision and Cognitive Services.

We used the following datasets:
- Imagenette: https://github.com/fastai/imagenette
- Pascal: https://deepai.org/dataset/pascal-voc
- Places: https://paperswithcode.com/dataset/places205
- Birds: https://www.kaggle.com/gpiosenka/100-bird-species
- Flowers: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Here you can find the pretrained models used in this project:
- Dahl: https://tinyclouds.org/colorize/ (Download section)
- Zhang eccv_16: https://github.com/richzhang/colorization/blob/master/colorizers/eccv16.py - related paper [here](https://arxiv.org/abs/1603.08511)
- Zhang siggraph17: https://github.com/richzhang/colorization/blob/master/colorizers/siggraph17.py - related paper [here](https://arxiv.org/abs/1705.02999)
- ChromaGAN: https://github.com/pvitoria/ChromaGAN - related paper [here](https://arxiv.org/abs/1907.09837)
- InstColorization: https://github.com/ericsujw/InstColorization - related paper [here](https://arxiv.org/abs/2005.10825)

## Requirements
- python 3.6 or 3.8
- virtualenv wrapper: https://virtualenvwrapper.readthedocs.io/en/latest/

## Dahl
- download the pretrained model and place the `colorize.tfmodel` file in the `models` folder
- create a virtual environment: `mkvirtualenv --python=python3 dahl`
- install the requirements: `pip install -r requirements_dahl.txt`
- position yourself into the following folder: `cd src/models`

- run the model: `python3 dahl.py`
    
## Eccv16 and Siggraph17
- create a virtual environment: `mkvirtualenv --python=python3 zhang`
- install the requirements: `pip install -r requirements_zhang.txt`
- position yourself into the following folder: `cd src/models`

- run the model: `python3 Eccv16andSiggraph17.py`

## ChromaGAN
- download the pretrained model and place the `ChromaGAN.h5` file in the `models` folder
- create a virtual environment using python 3.6: `mkvirtualenv --python=python3 chromaGAN`
- install the requirements: `pip install -r requirements_chromaGAN.txt`
- position yourself into the following folder: `cd src/models`

- run the model: `python3 chromaGAN.py`

## InstColorization
The code to run this model is contained in the following notebook: `src/models/InstColorization.ipynb`.
