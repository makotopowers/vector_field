import numpy as np
import matplotlib.pyplot as plt

# define the system of ODEs where 
# x' = f(x, y)
# y' = g(x, y)

def f(x, y):
    return x*(4-y-x**2)

def g(x, y):
    return y*(x-1)

# define the grid
x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 5, 30)

# create the grid
X, Y = np.meshgrid(x, y)

# compute the vector field
u = f(X, Y)
v = g(X, Y)

# for plotting, make all arrows the same length
norm = np.sqrt(u**2 + v**2)
u = u/norm
v = v/norm

def particle_sim(x_0, y_0):
    # simulate a particle moving in the vector field
    # start at [x_0, y_0]

    # define the time step
    dt = 0.001

    # define the number of steps
    n_steps = 20000

    # create the arrays to store the positions
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)

    # set the initial position
    x[0] = x_0
    y[0] = y_0

    # loop over the steps
    for i in range(n_steps-1):
        # compute the next position
        x[i+1] = x[i] + dt*f(x[i], y[i])
        y[i+1] = y[i] + dt*g(x[i], y[i])

    # assert that the particle stops
    assert np.abs(x[-1] - x[-2]) < 1e-10 and np.abs(y[-1] - y[-2]) < 1e-10
    return x, y

# get a bunch of points near (1,1)
x_0s = np.random.normal(1, 0.2, 50)
y_0s = np.random.normal(1, 0.2, 50)

# get a bunch of points near (2,2)
x_0s = np.append(x_0s, np.random.normal(2, 0.2, 50))
y_0s = np.append(y_0s, np.random.normal(2, 0.2, 50))

plots = []
for x_0, y_0 in zip(x_0s, y_0s):
    x, y = particle_sim(x_0, y_0)
    plots.append([x, y])

plt.figure(figsize=(10,10))
    

for plot in plots:
    plt.plot(plot[0], plot[1], 'k', alpha=0.3)
plt.grid()

# make the back of teh image white
plt.gca().set_facecolor('white')


# plot the vector field
plt.quiver(X, Y, u, v)

# plot [0,0], [1,3], [2,0], [-2,0]
plt.plot(0,0, 'ro')
plt.plot(1,3, 'ro')
plt.plot(2,0, 'ro')
plt.plot(-2,0, 'ro')

# plot the trajectory

plt.show()
