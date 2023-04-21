import os, json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import Utils
import numpy as np
from math import pi

def objectiveFunction(X):
    
    A = 10
    return A * 2 + sum([(x ** 2 - A * np.cos(2 * pi * x)) for x in X])

results = os.path.dirname(__file__) + "/Results"
executions = os.listdir(results)

for execution in executions:

    pop = json.load(open('{}/{}/population.json'.format(results, execution)))
    
    n_bits = 15
    iterations = 101
    functions = Utils(n_bits, -10, 10)
    
    dic = {}
    df = pd.DataFrame()
    
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylim(-10, 10)
    ax.set_ylabel('y', fontsize=20)
    feature_x = np.arange(-10, 11, 1)
    feature_y = np.arange(-10, 11, 1)
    [X, Y] = np.meshgrid(feature_x, feature_y)
    Z=objectiveFunction([X,Y])
    scat = ax.contour(X, Y, Z)
    
    for i in pop:
        dic["x"] = {}
        dic["y"] = {}
    
        for j in range(n_bits * 2):
            dic["x"][j] = functions.binToDec(pop[i][j][0:n_bits])
            dic["y"][j] = functions.binToDec(pop[i][j][n_bits:])
    
        tmp = pd.DataFrame(dic)
        tmp['iter'] = i
    
        df = pd.concat([df, tmp])
    
    def get_date(iteration):
        return df[df.iter == iteration]['x'], df[df.iter == iteration]['y']
    
    def animate(i):
        fig.clear()
        ax = fig.add_subplot(111, xlim=(-10, 10), ylim=(-10, 10))
        ax.set_title('Iteracao {}'.format(i), fontsize=20)
        x,y = get_date(str(i))
    
        feature_x = np.arange(-10, 11, 1)
        feature_y = np.arange(-10, 11, 1)
        [X, Y] = np.meshgrid(feature_x, feature_y)
        Z=objectiveFunction([X,Y])
        scat = ax.contour(X, Y, Z)
    
        scat = ax.scatter(x, y, c='blue')
        return scat
    
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=1000)
    
    anim.save('{}/{}/animation.gif'.format(results, execution), writer='pillow')

    plt.close()