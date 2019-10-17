print('program begins')
import numpy as np
import matplotlib.pyplot as plt

def function(x1,x2,y1,y2,z1,z2,N):
    alpha=2
    r1=np.sqrt(x1**2 + y1**2 + z1**2)
    r2=np.sqrt(x2**2 + y2**2 + z2**2)

    denominator = np.sqrt((x2-x1)*(x2-x1) +(y2-y1)*(y2-y1) + (z2- z1)*(z2- z1))

    return np.where(denominator < 1e-4,0, np.exp(-2*alpha*(r1 + r2)) /denominator)
    """
    if denominator < 1e-4:
        return 0

    else:
        return np.exp(-2*alpha*(r1 + r2)) /denominator
    """

def MonteCarloNormal(N,lmbda):

    a=-lmbda
    b= lmbda


    x1 = np.random.uniform(0,1,N)*(b-a) + a
    y1 = np.random.uniform(0,1,N)*(b-a) + a
    z1 = np.random.uniform(0,1,N)*(b-a) + a

    x2 = np.random.uniform(0,1,N)*(b-a) + a
    y2 = np.random.uniform(0,1,N)*(b-a) + a
    z2 = np.random.uniform(0,1,N)*(b-a) + a

    fx=function(x1,x2,y1,y2,z1,z2,N)

    crude_mc = np.sum(fx)/N

    integral=(crude_mc)*(b-a)**6
#    standard_dev = np.sqrt(np.sum(func_values - mean_value)**2)/(N)

#    total_mean_value = np.sum(mean_value)
#    total_standard_dev = np.sum(standard_dev)

    return integral#,standard_dev

integral=MonteCarloNormal(10000001,2.8)

print("Computed integral value: {}".format(integral))
#print("Standard Deviation: {}" .format(standard_deviation))
print("Actual integral value: {}".format(5*np.pi**2/16**2))
