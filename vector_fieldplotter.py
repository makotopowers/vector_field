import numpy as np
import matplotlib.pyplot as plt


# define the system of ODEs where 
# x' = f(x, y)
# y' = g(x, y)


class VectorField:

    vf = None


    def plot_vector_field_2d(self, x_min, x_max, y_min, y_max, x_step, y_step, show = False):
        # define the grid
        x = np.linspace(x_min, x_max, x_step)
        y = np.linspace(y_min, y_max, y_step)

        # create the grid
        X, Y = np.meshgrid(x, y)

        # compute the vector field
        u = self.f(X, Y)
        v = self.g(X, Y)
        

        # for plotting, make all arrows the same length
        norm = np.sqrt(u**2 + v**2)
        u = u/norm
        v = v/norm

        fig = plt.figure(figsize=(10,10))
        # background black 
        plt.gca().set_facecolor('black')
        plt.quiver(X, Y, u, v, color='green')
        
        if show:
            plt.show()

        self.vf = fig

  
    def particle_sim(self, x_0, y_0, dt, n_steps):
        # simulate a particle moving in the vector field
        # start at [x_0, y_0]

        # create the arrays to store the positions
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        

        # set the initial position
        x[0] = x_0
        y[0] = y_0
        

        # loop over the steps
        for i in range(n_steps-1):
            # compute the next position
            x[i+1] = x[i] + dt*self.f(x[i], y[i])
            y[i+1] = y[i] + dt*self.g(x[i], y[i])
            

        # assert that the particle stops
        assert np.abs(x[-1] - x[-2]) < 1e-10 and np.abs(y[-1] - y[-2]) < 1e-10
        return x, y
    
    def plot_particle_sim(self, startx, starty, dt, n_steps):
        x_0s = np.random.normal(startx, 0.5, 100)
        y_0s = np.random.normal(starty, 0.5, 100)

        fig = plt.figure(self.vf)

        # make the arrows green


        for x_0, y_0 in zip(x_0s, y_0s):
            x, y = self.particle_sim(x_0, y_0, dt, n_steps)

            plt.plot(x, y, linewidth=0.5, color = 'red', alpha = 1)
     
        # put x and y axes 
        plt.axhline(0, color='white')
        plt.axvline(0, color='white')

        plt.show()




    def f(self,x, y):
        return x*(4-y-x**2)

    def g(self, x, y):
        return y*(x-1)


def main():
    # define the vector field


    vf = VectorField()
    vf.plot_vector_field_2d(-5, 5, -5, 5, 30, 30)
    vf.plot_particle_sim(1, 2, 0.01, 10000)


main()