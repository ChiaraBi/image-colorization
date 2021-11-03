# Vision project
Project for the university course Vision and Cognitive Services.

We used a small subset of ImageNet, Pascal, Places and other datasets:
- Imagenette: https://github.com/fastai/imagenette
- https://github.com/EliSchwartz/imagenet-sample-images
- Pascal: https://deepai.org/dataset/pascal-voc
- Places: https://paperswithcode.com/dataset/places
- Birds: https://www.kaggle.com/gpiosenka/100-bird-species
- Flowers: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Here you can find the pretrained models used in this project:
- Dahl: https://tinyclouds.org/colorize/ (Download section)
- Zhang eccv_16: https://github.com/richzhang/colorization/blob/master/colorizers/eccv16.py
- Zhang siggraph17: https://github.com/richzhang/colorization/blob/master/colorizers/siggraph17.py
- ChromaGAN: https://github.com/pvitoria/ChromaGAN - related paper [here](https://arxiv.org/pdf/1907.09837.pdf)
- Su: https://github.com/ericsujw/InstColorization - related paper [here](https://arxiv.org/pdf/2005.10825.pdf)

Other recent pretrained models:
- https://github.com/LenKerr/Colorization-1 - inspired by Zhang
- https://github.com/dongheehand/MemoPainter-PyTorch - related paper [here](https://arxiv.org/pdf/1906.11888.pdf)
- https://github.com/MarkMoHR/Awesome-Image-Colorization - repository containing various codes/ideas (language-based image editing)

## Requirements
- python 3.6
- virtualenv wrapper: https://virtualenvwrapper.readthedocs.io/en/latest/

## Dahl
- download the pretrained model and place the `colorize.tfmodel` file in the `models` folder
- create a virtual environment: `mkvirtualenv --python=python3 dahl`
- install the requirements: `pip install -r requirements_dahl.txt`
- position yourself into the following folder: `cd src/models`

- run the model: 
  - to colorize black and white images: `python3 dahlBW.py`
  - to colorize ImageNet images: `python3 dahlImageNet.py`
    
## Zhang
- create a virtual environment: `mkvirtualenv --python=python3 zhang`
- install the requirements: `pip install -r requirements_zhang.txt`
- position yourself into the following folder: `cd src/models`

- run the model: 
  - to colorize black and white images: `python3 zhangBW.py`
  - to colorize ImageNet images: `python3 zhangImageNet.py`

## ChromaGAN
- download the pretrained model and place the `ChromaGAN.h5` file in the `models` folder
- create a virtual environment using python 3.6: `mkvirtualenv --python=python3 chromaGAN`
- install the requirements: `pip install -r requirements_chromaGAN.txt`
- position yourself into the following folder: `cd src/models`

- run the model: `python3 chromaGAN.py`
