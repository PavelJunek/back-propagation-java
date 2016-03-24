# Back propagation demo

A demo application for training a neural network and recognizing handwritten digits

This application is created for the students of CULS in Prague to demonstrate the training of neural network. The algorithm is quite slow, inefficient and not optimized, so it should not be used in production. The aim is to demonstrate the basic backpropagation algorithm.

The program runs from command line and requires two arguments: a file containing the training set and a file with the validation set. Both must be CSV files with 65 columns, where 64 columns describe the pixels of the image (8x8 pixels) and the last column contains the digit (0-9) displayed on the image.

The program trains the network using the training set and displays the error on the validation set after each training epoch. Then the user enters filename of another file (also CSV, but now only with 64 columns) and the program tries to classify the image encoded in the file.

You can experiment with the number of neurons in the hidden layer, the number of learning epochs and perhaps try different activation functions.

I hope it will be useful.
