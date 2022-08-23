# My neural network
### Author: Teodoro Sullazzo
<hr>
This is a simple implementation of a Neural Network in Java (a feedforward one). This was done primarily due to a moment of absolute boredom.
<br>
This work includes:

* A framework to create and customize the network.
* A framework to train the network, given the training data and the related labels.
* Some examples of use, with notebooks for the visualization of the network's errors.

The framework offers the opportunity to specify the size of the minibatches and the number of epochs
(I decided to implement the SGD with mini-batches strategy).

<br>

Further works will aim:
* ~~To adapt and change the learning rate dynamically.~~ $\leftarrow $ it was done with the momentum
* To add other optimizations, like parallelization using multithreading.

To prove that the ANN works, I included two examples of its use: the XOR function and the MNIST classification problem. In the first case, the model performed very well, even if the network was really small. In the second use case, the network achieved an accuracy of 90%. The ANN used in this case is saved in the "object.txt" file. This result seemed sufficient. Pushing further into the training phase could cause overfitting.

## Implementation's details

The forward propagation of the inputs is very simple. The interesting part refers to the backpropation, i.e. the learning phase.
<br>
The objective is to evaluate the derivative of the cost function with respect to weights and biases.
$$ \frac{\partial C}{\partial w_{ij}} $$

This value is obtained using the chain rule.

$$ \frac{\partial C}{\partial w_{ij}^l} = \frac{\partial C}{\partial a_j^l } \frac{\partial a_j^l }{\partial z_j^l }\frac{\partial z_j^l }{\partial w_{ij}^l}$$

Where

$$ \frac{\partial C}{\partial a_j^l }=a_k^{l-1}$$

For the biases ($\frac{\partial C}{\partial b_{j}^l}$), the $\frac{\partial C}{\partial a_j^l }$ part become equals to "1"

$$ \frac{\partial a_j^l }{\partial z_j^l } = \sigma^{'}(z_j^l) $$

And, lastly $ \frac{\partial z_j^l }{\partial w_{ij}^l}$ depends on whether the layer l is the last layer or if it's a hidden layer. <br>
For the last layer (referred to with a capital l, "L").
$$ \frac{\partial z_j^L }{\partial w_{ij} ^L} = 2(a_j^L-y_j)$$
Where $y_j$ is the real value and $a_j^L$ the predicted one.
<br>
For the hidden layers, the formula is.
$$ \frac{\partial z_j^l }{\partial w_{ij} ^l} = \sum_{j=0}^{n_{l+1}-1}w_{jk}^{l+1}\sigma^{'}(z_j^{l+1})\frac{\partial C}{\partial a_j^{l+1}} $$

All this stuff is implemented in the Layer class.


## Glossary
a := the activation of a neuron (i.e. the linear part after the application of the activation function).

z := the linear part of the neuron's output










