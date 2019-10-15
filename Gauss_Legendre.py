print('program begins')
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from sympy.functions import exp

"""
def Gauss_Legendre_exponential(N):
    a=np.zeros(N)
    a[-1]=1
    roots=np.polynomial.legendre.legroots(a)
    sorted_roots=np.sort(roots)
    func_values=np.exp(sorted_roots)
    n_list=np.linspace(0,N,N)
    legendre_list=np.zeros(N)
    index=0
    for i in n_list:
        legendre_list[index]=sy.special.legendre(i)
        index+=1
    for j in legendre_list:
        j()

    return 0
"""

def MonteCarloNormal(N):

    points=np.random.uniform(0,2,(N,6))
    func_values=np.exp(points)
    mean_value=np.mean(func_values)
    standard_dev=np.sqrt(np.sum(func_values-mean_value)**2 /N)
    return mean_value,standard_dev

def MonteCarloExponential(N):

    points=np.random.uniform(0,2,N)
    func_values=np.exp(points)
    mean_value=np.mean(func_values)
    standard_dev=np.sqrt(np.sum(func_values-mean_value)**2 /N)
    return mean_value,standard_dev


mean_value,standard_deviation=MonteCarloNormal(1000000)
print(2*mean_value)
print(standard_deviation)
