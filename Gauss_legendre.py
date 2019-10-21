print('program begins')
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

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

def polar_integrand(a1, b1, a2, b2, r1, r2, t1, t2, p1, p2):
    tol = 1e-10

    t1 = (b1-a1)*t1/2 + (b1+a1)/2
    t2 = (b1-a1)*t2/2 + (b1+a1)/2
    p1 = (b2-a2)*p1/2 + (b2+a2)/2
    p2 = (b2-a2)*p2/2 + (b2+a2)/2

    cosb = np.cos(t1)*np.cos(t2) + np.sin(t1)*np.sin(t2)*np.cos(p1-p2)

    r12 = r1**2+r2**2-2*r1*r2*cosb

    if r12 > tol:
        I = np.exp(-3*(r1+r2))*r1**2*r2**2*np.sin(t1)*np.sin(t2)/np.sqrt(r12)
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
    n = np.linspace(3,25,12)
    #n = np.linspace(3,7,3)
    Leg_I = np.zeros(len(n))
    lamb = 3
    time_spent_leg = np.zeros(len(n))


    for g in range(len(n)):
        N = int(n[g])
        xi, w = Gauss_leg(N)

        a = -lamb
        b = lamb
        start = time.time()
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        for m in range(N):
                            for h in range(N):
                                Leg_I[g] += w[i]*w[j]*w[k]*w[l]*w[m]*w[h]*integrand(a, b, xi[i], xi[j], xi[k], xi[l], xi[m], xi[h])
            print((i+1)/N*100)
        Leg_I[g] *= ((b-a)/2)**6
        end = time.time()
        time_spent_leg[g] = end-start

    print(Leg_I, 5*np.pi**2/16**2)
    plt.plot(n, Leg_I, 'bo-', label='Gauss Legendre for Cartesian')


    #  Laguerre part

    Lag_I = np.zeros(len(n))
    at = 0; bt = np.pi
    ap = 0; bp = 2 * np.pi
    time_spent_lag = np.zeros(len(n))

    for g in range(len(n)):
        N = int(n[g])
        ri, w_ri = Gauss_lag(N)
        print(N, ri, w_ri)
        angle, w_angle = Gauss_leg(N)

        start = time.time()
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        for m in range(N):
                            for h in range(N):
                                Lag_I[g] += w_ri[i] * w_ri[j] * w_angle[k] * w_angle[l] * w_angle[m] * w_angle[h] \
                                        * polar_integrand(at, bt, ap, bp, ri[i], ri[j], angle[k], angle[l], angle[m], angle[h])
            print((i+1)/N*100)
        Lag_I[g] *= ((bt - at) / 2) ** 2 * ((bp - ap)/2)**2
        end = time.time()
        time_spent_lag[g] = end-start

    plt.plot(n, Lag_I, 'ro-', label='Gauss Laguerre for radial')
    plt.axhline(y=5*np.pi**2/16**2)
    plt.xlabel('mesh points, N')
    plt.ylabel('integration value')
    plt.legend()
    plt.show()

    file = open('solution_GQ.txt', 'w')

    for i in range(len(n)):
        file.write('N=%g, Gauss_leg = %g, time = %g, Gauss_lag = %g, time = %g \n' % (n[i], Leg_I[i], time_spent_leg[i],
                                                                                      Lag_I[i], time_spent_lag[i]))
    print(n)
    print(time_spent_leg)
    print(time_spent_lag)