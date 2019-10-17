import numpy as np
import matplotlib.pyplot as plt

def spherical_function(r1,r2,theta1,theta2,phi1,phi2,N):
    alpha=2
    cosbeta=np.cos(theta1)*np.cos(theta2) + np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2)

    denominator = np.sqrt(r1**2 + r2**2 - 2*r1*r2*cosbeta)

    return np.where(denominator < 1e-4, 0, (np.exp(-2*alpha*(r1 + r2)) * r1**2 * r2**2 *np.sin(theta1)*np.sin(theta2)) / denominator)

def MonteCarlo(N):
    lmbda=2.8
    a=-lmbda
    b=lmbda

    A = np.ones(N)*a

    r1 = np.random.exponential(1,N)*(b-a) + a
    r2 = np.random.exponential(1,N)*(b-a) + a
    theta1 = np.random.exponential(1,N)*(np.pi)# + a
    theta2 = np.random.exponential(1,N)*(np.pi)# + a
    phi1 = np.random.exponential(1,N)*(2*np.pi)# + a
    phi2 = np.random.exponential(1,N)*(2*np.pi)# + a

    fx=spherical_function(r1,r2,theta1,theta2,phi1,phi2,N)

    crude_mc = np.sum(fx)/N

    integral=(crude_mc)*(b-a)**6

    return integral

integral=MonteCarlo(100001)

print(integral)
