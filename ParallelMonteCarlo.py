import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
"""
def spherical_function(r1,r2,theta1,theta2,phi1,phi2,N):
    alpha=2
    #changing to spherical coordinates
    cosbeta=np.cos(theta1)*np.cos(theta2) + np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2)
    denominator = np.sqrt(r1**2 + r2**2 - 2*r1*r2*cosbeta)
    #returning values only when denominator < 10^-4
    return np.where(denominator < 1e-4, 0,   (r1**2 * r2**2 *np.sin(theta1)*np.sin(theta2)) / denominator)
"""

def MonteCarlo(N):

    #expo_scale=4 # scaling the random exponential distribution to accomodate 6D

    #making random exponential distribution from scratch
    r12 = np.random.uniform(0,1,N)
    r22 = np.random.uniform(0,1,N)
    r1 = -0.25*np.log(1-r12)#np.random.exponential(1/expo_scale,N)
    r2 = -0.25*np.log(1-r22)#np.random.exponential(1/expo_scale,N)

    #angles have random normal distribution
    theta1 = np.random.uniform(0,1,N)*(np.pi)# + a
    theta2 = np.random.uniform(0,1,N)*(np.pi)# + a
    phi1 = np.random.uniform(0,1,N)*(2*np.pi)# + a
    phi2 = np.random.uniform(0,1,N)*(2*np.pi)# + a
    alpha=2
    #changing to spherical coordinates
    cosbeta=np.cos(theta1)*np.cos(theta2) + np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2)
    denominator = np.sqrt(r1**2 + r2**2 - 2*r1*r2*cosbeta)
    #returning values only when denominator < 10^-4
    spherical_function = np.where(denominator < 1e-4, 0,   (r1**2 * r2**2 *np.sin(theta1)*np.sin(theta2)) / denominator)

    lmbda=2.8
    a=-lmbda
    b=lmbda

    fx=spherical_function

    crude_mc = np.sum(fx)/N

    argument = (fx - np.mean(fx))**2

    variance = np.sum( argument ) /len(fx) * (4*np.pi**4/16)

    integral= crude_mc*(4*np.pi**4 )/16 #accomodating for 6D spherical

    return integral, variance
start=time.time()
integral,variance=MonteCarlo(1_00_00_00_0)
end=time.time()
exact=(5*np.pi**2)/16**2
print("Computed integral: {:.3f}".format(integral))
print("Variance: {}".format(variance))
print("Time to compute integral: {:.3f}".format(end-start))
print("Exact Integral: {:.3f} ".format(exact))

N=1_00_0
N_list=np.full(1_00_00,N)

def main():
    number_of_processors=mp.cpu_count()
    pool = mp.Pool(number_of_processors)
    result= np.array(pool.map( MonteCarlo, N_list ))
    result_int,result_var=result.T
    print("Integral calculated using parallelization: {:.3f}".format(np.mean(result_int)))
    print("Variance calculated using parallelization: {}".format(np.mean(result_var)))
start2=time.time()
main()
end2=time.time()
print("Time to compute parallell integral: {:.3f}".format(end2-start2))
