LNN
===

LNN (Little Neural Network) is a command-line C program running, training, and testing feedforward neural networks, with the following features.

- Light weight, containing only a standalone executable;
- Serve as a Unix filter; Easy to work with other programs;
- Plain-text formats of models, input, output, and samples;
- Compact notations;
- Different activation functions for different layers;
- L2 regularization;
- Mini-batch training.

**Table of Contents**

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Further Documentation](#further-documentation)

Installation
------------

It would be better to select a version from the [release page](https://github.com/dongyx/lnn/releases)
than downloading the working code,
unless you understand the status of the working code.
The latest stable release is always recommended.

	$ make
	$ sudo make install

By default, LNN is installed to `/usr/local`.
You could call `lnn --version` to check the installation.

Getting Started
---------------

The following call of LNN creates a network with
a 10-dimension input layer,
a 5-dimension hidden layer,
and a 2-dimension output layer.

	$ lnn train -C q10i5s2s samples.txt >model.nn

The `-C` option creates a new model with the structure specified by the argument.
The argument here is `q10i5s2s`.
The first character `q` specifies the loss function to be the quadratic error.
The following three strings `10i`, `5s`, `2s` represent that
there are 3 layers,
including the input layer,
with dimensions 10, 5, 2, respectively.
The character following each dimension specifies the activation function for that layer.
Here `i` and `s` represent the identity function and the sigmoid function respectively ([Further Documentation](#further-documentation)).

The remaining part of this chapter assumes that
the network maps $R^n$ to $R^m$.
In words, it has a $n$-dimension input layer and $m$-dimension output layer.

LNN reads samples from the file operand, or, by default, the standard input. 
The trained model is printed to the standard output in a text format.

The sample file is a text file containing numbers separated by white characters (space, tab, newline).
Each $n+m$ numbers constitute a sample.
The first $n$ numbers of a sample constitute the input vector,
and the remaining constitute the output vector.

LNN supports many training arguments like learning rate, iteration count, and batch size ([Further Documentation](#further-documentation)).

LNN could train a network based on an existed model
by replacing `-C` with `-m`.

	$ lnn train -m model.nn samples.txt >model2.nn

This allows one to observe the behaviors of the model in different stages
and provide different training arguments.

The `run` sub-command runs an existed model.

	$ lnn run -m model.nn input.txt

LNN reads the input vectors from the file operand, or, by default, the standard input. 
The input shall contain numbers separated by white characters
(space, tab, newline).
Each $n$ numbers constitute an input vector.

The output vector of each input vector is printed to the standard output.
Each line contains an output vector.
Components of an output vector are separated by a space.

The `test` sub-command evaluates an existed model.

	$ lnn test -m model.nn samples.txt

LNN reads samples from the file operand, or, by default, the standard input. 
The mean loss value of the samples is printed to the standard output.
The format of the input file is the same as of the `train` sub-command.

Further Documentation
---------------------

- The [technical report](https://www.dyx.name/notes/lnn.html) serves as an extension of this read-me file.
It contains more details and examples for understanding the design and usage.

- Calling `lnn --help` prints a brief of the command-line options.
