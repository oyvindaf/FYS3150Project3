print('program begins')
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time


def function(x1,x2,y1,y2,z1,z2,N): #Definging the function to be integrated
    alpha=2
    r1=np.sqrt(x1**2 + y1**2 + z1**2)
    r2=np.sqrt(x2**2 + y2**2 + z2**2)

    denominator = np.sqrt((x2-x1)*(x2-x1) +(y2-y1)*(y2-y1) + (z2- z1)*(z2- z1))
    #Returning only the function values where denominator <10^4
    return np.where(denominator < 1e-4,0, np.exp(-2*alpha*(r1 + r2)) /denominator)

def MonteCarloNormal(N,lmbda):
    #integral limits
    a=-lmbda
    b= lmbda

    #Making uniform and random distribution of N points from -lambda to lambda for all variables
    x1 = np.random.uniform(a,b,N)#*(b-a) + a
    y1 = np.random.uniform(a,b,N)#*(b-a) + a
    z1 = np.random.uniform(a,b,N)#*(b-a) + a

    x2 = np.random.uniform(a,b,N)#*(b-a) + a
    y2 = np.random.uniform(a,b,N)#*(b-a) + a
    z2 = np.random.uniform(a,b,N)#*(b-a) + a

    #allocating constant to function()
    fx=function(x1,x2,y1,y2,z1,z2,N)

    crude_mc = np.sum(fx)/N #summing

    integral=(crude_mc)*(b-a)**6 #accounting for 6 dimensions by multiplying with (b-a)^6
    argument = (fx - np.mean(fx))**2
    variance = np.sum( argument ) / N * (b-a)**6

#    total_mean_value = np.sum(mean_value)
#    total_standard_dev = np.sum(standard_dev)

    return integral, variance



N_iteration_list = np.array([1_00,1_00_0,1_00_00,1_00_00_0,1_00_00_00,1_00_00_00_0,1_00_00_00_00])

integral_list=[]
variance_list=[]
normal_time_list=[]

iterations=1
for N in N_iteration_list:


    start=time.time()
    integral,variance=MonteCarloNormal(N,2.8)
    end=time.time()


    integral_list.append(integral)
    variance_list.append(variance)
    normal_time_list.append(end-start)


    print("Iterations: {}".format(iterations))
    print("N = {}".format(N))
    print("Integral value: {}".format(integral))

    iterations +=1
"""
plt.plot(N_iteration_list[1:],integral_list[1:])
plt.scatter(N_iteration_list[1:],integral_list[1:])
plt.xlabel("Number of iterations",fontsize=16)
plt.ylabel("Integral value",fontsize=16)
plt.title("Integral values for brute force Monte Carlo",fontsize=16)
plt.tight_layout()
plt.savefig("brute_int.pdf")
plt.close()


plt.plot(N_iteration_list[1:],variance_list[1:])
plt.scatter(N_iteration_list[1:],variance_list[1:])
plt.xlabel("Number of iterations",fontsize=16)
plt.ylabel("Variance",fontsize=16)
plt.title("Variance as a function of N",fontsize=16)
plt.tight_layout()
plt.savefig("brute_var.pdf")
plt.close()


plt.plot(N_iteration_list,normal_time_list)
plt.scatter(N_iteration_list,normal_time_list)
plt.xlabel("Numer of iterations",fontsize=16)
plt.ylabel("Time in seconds",fontsize=16)
plt.title("Time to brute force compute Monte Carlo Integral",fontsize=16)
plt.tight_layout()
plt.savefig("brute_time.pdf")
plt.close()
"""
