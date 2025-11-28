import numpy as np
import networkx as nx
from scipy import integrate


# 原始系统
def func_neuronal(t, y, f, g1, mu, delta, A):
    dydt = -f * y + g1 * np.dot(A, (1 / (1 + np.exp(mu - delta * y)) - 1 / (1 + np.exp(mu))))
    return dydt


def func_grn(t, y, f, g1, a, h, A):
    dydt = -f * (y ** a) + g1 * np.dot(A, (y ** h / (1 + y ** h)))
    return dydt

# SIS动力学
def func_sis(t,y,f,g1,A):
    dydt = f*y + g1*np.dot(A,y)*(1-y)
    return dydt

# double_well
def func_double_well(t,y,r1,r2,r3,g1,A):
    dydt = -(y-r1)*(y-r2)*(y-r3) + g1*np.dot(A,y)
    return dydt


def sis(args):
    f,g,graph, initial = args
    t0 = 0
    Adj = nx.to_numpy_array(graph)
    y0 = initial #* np.ones(np.shape(Adj)[0])
    termination = True
    m_s = 0.01
    while termination:
        t_max = t0 + 5
        t_span = (t0, t_max)
        result1 = integrate.solve_ivp(func_sis,
                             t_span,
                             y0,
                             method='RK45',args=(f, g,Adj),
                             dense_output=True,
                             max_step=m_s)
        error = np.max(np.abs((result1.y[:, -1] - result1.y[:, -2])/(m_s)))
        if ( error < 1e-7) or (t_max>500):
            termination = 1 - termination
        else:
            t0 = t_max
            y0 = result1.y[:, -1]

    # error = np.max(np.abs(result1.y[:, -1] - result1.y[:, -2]))/m_s
    return [termination,error,t_max,g,result1.y[:, -1]]


def double_well(args):
    r1,r2,r3,g1,graph, initial = args
    t0 = 0
    Adj = nx.to_numpy_array(graph)
    y0 = initial #* np.ones(np.shape(Adj)[0])
    termination = True
    m_s = 0.01
    while termination:
        t_max = t0 + 5
        t_span = (t0, t_max)
        result1 = integrate.solve_ivp(func_double_well,
                             t_span,
                             y0,
                             method='RK45',args=(r1,r2,r3, g1, Adj),
                             dense_output=True,
                             max_step=m_s)
        error = np.max(np.abs((result1.y[:, -1] - result1.y[:, -2])/(m_s)))
        if ( error < 1e-7) or (t_max>500):
            termination = 1 - termination
        else:
            t0 = t_max
            y0 = result1.y[:, -1]

    # error = np.max(np.abs(result1.y[:, -1] - result1.y[:, -2]))/m_s
    return [termination,error,t_max,g1,result1.y[:, -1]]

