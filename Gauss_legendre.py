print('program begins')
import numpy as np
import matplotlib.pyplot as plt


def function(x):
    #alpha = 2
    #return np.exp(-2*alpha*x)
    return 3*x**4

def Gauss_leg(n):
    L_N = np.zeros(n+1)
    L_N[-1] = 1
    xi = np.polynomial.legendre.legroots(L_N)

    p1 = 1
    p2 = 0
    for j in range(1,n+1):
        p3 = p2
        p2 = p1
        p1 = ((2*j-1)*xi*p2 - (j-1)*p3)/j

    pp = n*(xi*p1-p2)/(xi**2-1)
    w = 2/((1-xi**2)*pp**2)

    return xi, w

if __name__ == '__main__':
    n = 3
    xi, w = Gauss_leg(n)

    Leg_I = 0
    for i in range(n):
        Leg_I += w[i]*function(xi[i])
    print(Leg_I)
