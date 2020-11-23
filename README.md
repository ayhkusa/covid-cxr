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


Section I - Replicate LIME in https://github.com/aildnont/covid-cxr

To Do: First you find this this implementation of the method called Local Interpretable Model-Agnostic Explanations (i.e. LIME). You also read this article and you get your hands dirty and replicate the results in your colab notebook with GPU enabled kernel(40%).

Alan's work: convert lime_explain.py to jupyter notebook
Notebook: https://github.com/ayhkusa/covid-cxr/blob/master/src/interpretability/lime_explain.ipynb
