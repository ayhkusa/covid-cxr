# CS 677 Deep Learning - Project #2 Explainable COVID-19 Pneumonia
### Prof. Pantelis Monogioudis
### By Student Alan Yeung

## System Setup

AWS instance type: g4dn.2xlarge, 8 cpu, 32 GB, 1 T4 gpu, 16 GB

OS Linux 18.04.1-Ubuntu

source activate tensorflow2_latest_p37

export PATH=/opt/tljh/user/bin:$PATH

## Project Github

https://github.com/ayhkusa/covid-cxr


## Section I - Replicate LIME in https://github.com/aildnont/covid-cxr

To Do: First you find this this implementation of the method called Local Interpretable Model-Agnostic Explanations (i.e. LIME). You also read this article and you get your hands dirty and replicate the results in your colab notebook with GPU enabled kernel(40%).

### Alan's work: convert lime_explain.py to jupyter notebook

Notebook: https://github.com/ayhkusa/covid-cxr/blob/master/src/interpretability/lime_explain.ipynb

Run results: https://github.com/ayhkusa/covid-cxr/blob/feature/alan/src/interpretability/Part%20I%20-%20LIME%20Model%20run%20results.pdf

### Challenges:

1. Outdated setup with recent version of tensorflow: [lime_explain.ipynb] need to change x as type double (predict_and_explain(x.astype('double'))

## Section II - Add SHAP to existing x-ray model in https://github.com/aildnont/covid-cxr

### Alan's work: convert train.py to jupyter notebook

Notebook: https://github.com/ayhkusa/covid-cxr/blob/feature/alan/src/train_shap2.ipynb

Run results: https://github.com/ayhkusa/covid-cxr/blob/feature/alan/src/Part%20II%20-%20SHAP%20Model%20run%20results.pdf

### Challenges:

1. Outdated setup with recent version of tensorflow:
    a. model.fit_generate to model.fit
    b. model.evaluate_generator to model.evaluate
    c. original x-ray model uses dcnn_resnet and SHAP example uses VGG. Needed to modify the example to use 
