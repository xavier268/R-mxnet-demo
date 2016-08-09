# R - MxNet demo

This application demonstrates a few ill documented features from the R mxnet package for neural networks, in particular :

* retraining an already partially trained model
* using an iterator in a multi-class context

The Neural Network is not optimized for the prediction implemented (predict a byte from the previous bytes). By default, it uses an utf-8 text (Text from Victor Hugo, in French) for training, ignoring line formatting bytes.
