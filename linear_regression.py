import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Building the Data Model
number_of_points = 200  # Number of points we want to draw 

# Initialize two lists, x_points and y_points. These lists will contain the generated points
x_point = []
y_point = []

# These two constants will appear in the linear relation of y with x.
a = 0.50
b = 0.22
#Via NumPy's random.normal function, we generate 200 random points around the regression equation y = 0.50x + 0.22:
for i in range(number_of_points):
    x = np.random.normal(0.0,0.5)
    y = a*x + b +np.random.normal(0.0,0.1)
    x_point.append([x])
    y_point.append([y])

plt.plot(x_point,y_point, 'o', label='Input Data')
plt.legend()
plt.show()

A = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
B = tf.Variable(tf.zeros([1]))
y = A * x_point + B

# We are using a mean square error cost function
# To minimize the cost function, we use an optimixation algorithm with the gradient descent

cost_function = tf.reduce_mean(tf.square(y - y_point))
optimizer = tf.train.GradientDescentOptimizer(0.2)  # learning rate is 0.2

#The learning rate determines how fast or slow we move towards the optimal weights. If it is very large, we skip the optimal solution, and if it is too small, we need too many iterations to converge to the best values.

train = optimizer.minimize(cost_function)
model = tf.global_variables_initializer()
with tf.Session() as session:
        session.run(model)
        for step in range(0,26):
                session.run(train)
                if (step % 5) == 0:
                        plt.plot(x_point,y_point,'o',
                                 label='step = {}'
                                 .format(step))
                        plt.plot(x_point,
                                 session.run(A) * 
                                 x_point + 
                                 session.run(B))
                        plt.legend()
                        plt.show()
