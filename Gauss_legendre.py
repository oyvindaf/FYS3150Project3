print('program begins')
import numpy as np
import matplotlib.pyplot as plt


def integrand(a, b, x1, x2, y1, y2, z1, z2):
    tol = 1e-10
    x1 = (b-a)*x1/2 + (b+a)/2
    x2 = (b-a)*x2/2 + (b+a)/2
    y1 = (b-a)*y1/2 + (b+a)/2
    y2 = (b-a)*y2/2 + (b+a)/2
    z1 = (b-a)*z1/2 + (b+a)/2
    z2 = (b-a)*z2/2 + (b+a)/2

    r1 = np.sqrt(x1**2+y1**2+z1**2)
    r2 = np.sqrt(x2**2+y2**2+z2**2)

    distance = np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2))

    if distance > 0:
        I = np.exp(-4*(r1+r2))/distance
    else:
        I = 0
    return I

def Gauss_leg(n):
    L_N = np.zeros(n+1)
    L_N[-1] = 1
    xi = np.polynomial.legendre.legroots(L_N)

    p1 = 1
    p2 = 0
    for j in range(n):
        p3 = p2
        p2 = p1
        p1 = ((2*j+1)*xi*p2 - j*p3)/(j+1)

    pp = n*(xi*p1-p2)/(xi**2-1)
    w = 2/((1-xi**2)*pp**2)

    return xi, w

def Gauss_lag(n):
    L_N = np.zeros(n+1)
    L_N[-1] = 1
    xi = np.polynomial.laguerre.lagroots(L_N)

    p1 = 1
    p2 = 0
    for j in range(n+1):
        p3 = p2
        p2 = p1
        p1 = ((2*j+1-xi)*p2 - j*p3)/(j+1)

    w = xi/((n+1)**2*p1**2)

    return xi, w

if __name__ == '__main__':
    n = 15
    lamb = 3
    xi, w = Gauss_leg(n)

    #  xi_new = np.tan(np.pi/4*(1+xi))
    #  w_new = np.pi/4*w/(np.cos(np.pi/4*(1+xi))**2)

    a = -lamb
    b = lamb
    Leg_I = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    for m in range(n):
                        for h in range(n):
                            Leg_I += w[i]*w[j]*w[k]*w[l]*w[m]*w[h]*integrand(a, b, xi[i], xi[j], xi[k], xi[l], xi[m], xi[h])
        print(i)
    Leg_I *= ((b-a)/2)**6
    print(Leg_I, 5*np.pi**2/16**2)
