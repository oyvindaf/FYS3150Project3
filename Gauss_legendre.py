print('program begins')
import numpy as np
import matplotlib.pyplot as plt


def function(a, b, x):
    #alpha = 2
    #return np.exp(-2*alpha*x)
    t = (b-a)*x/2 + (b+a)/2
    return np.exp(-2*t)

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

def Test_monte1d(b, a, n):
    x = np.random.uniform(a,b,n)
    fx = function(x)
    crude_mc = sum(fx)
    sigma = sum(fx*fx)

    integral = crude_mc*(b-a)/n
    return integral, sigma

if __name__ == '__main__':
    n = 100
    lamb = 3
    xi, w = Gauss_leg(n)
    xi_new = np.tan(np.pi/4*(1+xi))
    w_new = np.pi/4*w/(np.cos(np.pi/4*(1+xi))**2)


    start = -lamb; end = lamb
    Leg_I = 0
    for i in range(n):
        Leg_I += w[i]*function(start, end, xi[i])
    Leg_I *= (end-start)/2
    print(Leg_I)

    #print(Test_monte1d(2, 0, 1000000))
