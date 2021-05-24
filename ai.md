# Artificial intelligence: example of neuronal network in Python

## Description
Artificial neuronal networks are one of the most commonly used computational models to implement artificial intelligence. They consist of a set of units, called neurons, connected to each other to transmit signals.

Each neuron is connected to other neurons by links. Through them, it receives information from other neurons, evaluates it, and as it informs other neurons of the outcome of that evaluation, which can be so much to highlight a characteristic and to attenuate it.

## Objective
We will create a neuronal network with which we will examine cars and predict the distance you can go through every gallon of fuel.

First, some concepts and lue will be explained

## Structure
Commonly neuronal networks are implemented in the form of layers. The first layer, called input layer or layer 0 represents the different features that are evaluated from an object.

In our example, the first layer (also called layer 0) will be represented by the characteristics we will evaluate from the car. For example: the amount of cylinders, weight, acceleration, year and manufacturing origin, etc.

Then we will add two more layers, hidden calls, which will be responsible for evaluating the information, the relationship between the different characteristics and issuing a result to the posterior layer.

Finally we will have an output layer, which will obtain the previous hidden layer results and calculating the distance that the car can travel.

## Neurons
Each layer will be formed by several neurons, except the last layer, the output layer, which will have only one neuron.

Each neuron will be represented by the following formula:

$$\Large n = \sum W X + b$$

Where $\Large X$ is a vector (one-dimensional matrix) with the information coming from each one the neurons of the previous layer. In the case of the first hidden layer, the information you will receive will be directly the characteristics of the car.

$\Large W$ is a vector that will assign a weight at each value from the neurons of the previous layer. These pesos will be those that will generate an answer that store to get the final prediction.

$\Large b$ is a value that will apply an offset (BIAS) to the result.

## Non-linear Activation
The above function is a linear function. If all the neurons were linear functions, the result of the neuronal network would also be another linear function. This would not allow the network to identify complex relationships.

To avoid this, a non-linear component is added to each neuron, called activation function. In our case we will use the RELU function (rectified linear unit), which is nothing other than:

$$\Large a = ReLU( n ) = max( 0, n )$$

So the complete formula of each neuron, except for that of the output layer, it will be:

$$\Large a = max( 0, \sum W X + b )$$

## Final Linear Activation
As our goal is to predict a value (regression), instead detecting or classifying an object (classification), the last neuron of the network, that of the output layer, must generate a linear value.

The formula of this neuron will be like:

$$
\Large A = \sum W X + b
$$

## Optimization of the Neural Network: Learning

The network learning process will involve searching values ​​for the weights we use, so that when you feed it with the data of a car, the network produces the result we want to estimate: the journey through Galon.

**Steps:**

1. Initialize the weights we will use in our network in a random way and with a distribution that facilitates the calculations.
2. Repeat the following:
    1. Feed the network with the characteristics of a car.
    2. Perform the calculations of the neuronal network, multiplying the weights by the input values, adding the movements and applying the layer activation functions by layer, until the final result.
    3. Calculate the error (difference) between the estimate of the network and the real value, which we will call $\Large J (W, b)$
    4. From behind forward, using the difference obtained, we calculate the derivatives for the weights $\Large W$ and the offset $\Large b$ using the following formulas:

    $$\Large \frac{\partial J(W, b)}{\partial W}  and  \frac{\partial J(W, b)}{\partial b}$$

    5. With the derivatives, we proceed to modify the weights to bring them closer to the point where the difference reaches a minimum. For this, we multiply the derivative by an $\Large \alpha$ value (called Learning Rate) and we will subtract it to LS corresponding pesos:

    $$\Large W = W - \alpha * \frac{\partial J(W, b)}{\partial W}$$

    $$\Large b = b - \alpha * \frac{\partial J(W, b)}{\partial b}$$

    6. With the updated weights, the previous steps are repeated until the error, the difference between the predictions of the network and the actual values, is acceptably small.


## The Code

In practice, the optimization of the neuronal network of an example at a time is not performed. But they take advantage of code libraries that allow the capabilities of the CPU, GPU and TPU modern to perform calculations simultaneously on many examples.

In our case, we will use the Python language and mainly the Numpy library to perform the calculations on all examples simultaneously and obtain a better performance.

### The Model

We will define the model of our neuronal network.
