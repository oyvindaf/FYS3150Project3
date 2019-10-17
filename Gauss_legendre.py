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

    if distance > tol:
        I = np.exp(-4*(r1+r2))/distance
    else:
        I = 0
    return I

def Gauss_leg_numpy(n):
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
    n = np.linspace(3,15,7)
    Leg_I = np.zeros(len(n))
    lamb = 3

    '''
    for g in range(len(n)):
        N = int(n[g])
        xi, w = Gauss_leg(N)

        #  xi_new = np.tan(np.pi/4*(1+xi))
        #  w_new = np.pi/4*w/(np.cos(np.pi/4*(1+xi))**2)

        a = -lamb
        b = lamb

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        for m in range(N):
                            for h in range(N):
                                Leg_I[g] += w[i]*w[j]*w[k]*w[l]*w[m]*w[h]*integrand(a, b, xi[i], xi[j], xi[k], xi[l], xi[m], xi[h])
            print(i)
        Leg_I[g] *= ((b-a)/2)**6

    print(Leg_I, 5*np.pi**2/16**2)
    plt.plot(n, Leg_I)
    plt.show()
    '''

    xi, wi = Gauss_lag(5)
    print(xi, wi)
