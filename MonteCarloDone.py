print('program begins')
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


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
    print(argument)
    variance = np.sum( argument ) / N * (b-a)**6

#    total_mean_value = np.sum(mean_value)
#    total_standard_dev = np.sum(standard_dev)

    return integral, variance

integral,variance=MonteCarloNormal(1_00_00_00,2.8)

print("Computed integral value: {}".format(integral))
print("Variance: {}".format(variance))
#print("Standard Deviation: {}" .format(standard_deviation))
print("Actual integral value: {}".format(5*np.pi**2/16**2))
