##### Import libraries, numpy isn't very important for this particular NN
import numpy
from matplotlib import pyplot

##### Initialise the variables that will be constant throughout the program
data = [[3,1.5,1],[2,1,0],[4,1.5,1],[3,1,0],[3.5,0.5,1],[2,0.5,0],[5.5,1,1],[1,1,0]]
w1 = numpy.random.randn()
w2 = numpy.random.randn()         # Init the weights and bias as random floats
b = numpy.random.randn()

##### This is the neural network, it takes two inputs and produces one output without any hidden layers
def NN(length, width):
    return sigmoid(w1 * length + w2 * width + b)

##### Squishes the NN output into a number between 0 and 1
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

##### Find the amount of error between the prediction and the desired output
def cost(x, t):
    return (x - t) ** 2

##### The slope of the curve of the cost
def slope(x, t):
    return 2 * (x - t)

##### Simplify the error into either red or blue (0 for blue 1 for red)
def output(x):
    if x > 0.5:
        return 1
    else:
        return 0

##### Train the NN so it can accurately predict the typr of flower based on the length and width of its petals
def train():
    global w1, w2, b         # Call global so the global variables can be modified from within the function
    alpha = 0.2                  # Define the learning rate, the rate at which the slope is decended
    errors = []                    
    print('training', end='')
    for i in range(500000):            # Train the network 500000 times
        j = numpy.random.randint(len(data))                   # Pick a random flower
        if i % 50000 == 0:
            print('.', end='')
        error = cost(NN(data[j][0], data[j][1]), data[j][2])
        errors.append(error)                                    # Calculate the error for this flower and add it to the list 

        ##### Gradient decent
        dcdnn = slope(NN(data[j][0], data[j][1]), data[j][2])  # The derivative of the cost with respect to the NN
        dnndz = NN(data[j][0], data[j][1]) * (1 - NN(data[j][0], data[j][1])) # The derivative of the NN with respect to the NN's function (w1 * length + w2 * width + b) named z
        dcdz = dcdnn * dnndz # Multiply the previous two to get cost with respect to z
        # Sidenote the derivatives to the weights are the inputs (data[j][0/1]) and the bias is just 1
        w1 -= alpha * dcdz * data[j][0]   # Move w1 down the slope by the learning rate times  dcdw1
        w2 -= alpha * dcdz * data[j][1]   # Do the same to w2 and the bias
        b -= alpha * dcdz  * 1
    print('\n')
    pyplot.plot(errors)
    pyplot.show()

train()
while True: 
    print('Flower type predictor')
    length = float(input('Length: '))
    width = float(input('Width: '))               # Train the network and then let it make some predictions
    print('Prediction: ', end = '')
    if output(NN(length, width)) == 1:
        print('Red')
    else:
        print('Blue')
    ex = input('Do you want to exit [y/n]: ')
    if ex == 'y':
        break
