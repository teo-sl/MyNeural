# My neural network
## Teodoro Sullazzo
This is a simple implementation of a Neural Network in Java. The main objective of this work was to understand how this model works.
<br>
This work includes:
* A framework to create and customize the network.
* A framework to train the network, given the training data with the related labels.

The framework offers the opportunity to specify the size of the minibatches and the number of epochs
(I decided to implement the SGD with mini-batches strategy).

<br>

Further works will aim:
* To adapt and change the learning rate dynamically.
* To add other optimizations, like parallelization using multithreading.

To prove that the ANN works, I included two examples of its use: the XOR function and the MNIST classification problem. In the first case, the model performed very well, even if the network was really small. In the second use case, it was achieved that it had an accuracy of about 90%. The ANN used in this case was saved in the "object.txt" file. This result seemed sufficient. Pushing further into the training phase could cause overfitting.

