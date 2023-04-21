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
    m_pop = json.load(open('{}/{}/population_after_mut.json'.format(results, execution)))

    n_bits = 15
    iterations = 101
    functions = Utils(n_bits, -10, 10)

    dic = {}
    df = pd.DataFrame()
    m_dic = {}
    m_df = pd.DataFrame()

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
        m_dic["x"] = {}
        m_dic["y"] = {}

        for j in range(n_bits * 2):
            dic["x"][j] = functions.binToDec(pop[i][j][0:n_bits])
            dic["y"][j] = functions.binToDec(pop[i][j][n_bits:])
            m_dic["x"][j] = functions.binToDec(m_pop[i][j][0:n_bits])
            m_dic["y"][j] = functions.binToDec(m_pop[i][j][n_bits:])

        tmp = pd.DataFrame(dic)
        tmp['iter'] = i
        m_tmp = pd.DataFrame(m_dic)
        m_tmp['iter'] = i

        df = pd.concat([df, tmp])
        m_df = pd.concat([m_df, m_tmp])

    def get_date(iteration, dataset):
        if dataset == 0:
            return df[df.iter == iteration]['x'], df[df.iter == iteration]['y']
        else:
            iteration = iteration.split(".")[0]
            return m_df[m_df.iter == iteration]['x'], m_df[m_df.iter == iteration]['y']

    def animate(i):
        j = i % 1
        fig.clear()
        ax = fig.add_subplot(111, xlim=(-10, 10), ylim=(-10, 10))
        ax.set_title('Iteracao {}'.format(i), fontsize=20)
        x,y = get_date(str(i), j)

        feature_x = np.arange(-10, 11, 1)
        feature_y = np.arange(-10, 11, 1)
        [X, Y] = np.meshgrid(feature_x, feature_y)
        Z=objectiveFunction([X,Y])
        scat = ax.contour(X, Y, Z)

        scat = ax.scatter(x, y, c='red')
        return scat

    list = [x for x in range(101)]
    new_list = [0.1 + x for x in list]
    f_list = list + new_list
    f_list.sort()

    anim = animation.FuncAnimation(fig, animate, frames=f_list, interval=1000)

    anim.save('{}/{}/animation.gif'.format(results, execution), writer='pillow')

    plt.close()