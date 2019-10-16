print('program begins')
import numpy as np
import matplotlib.pyplot as plt

def function(x1,x2,y1,y2,z1,z2,N):
    alpha=2
    r1=np.sqrt(x1**2 + y1**2 + z1**2)
    r2=np.sqrt(x2**2 + y2**2 + z2**2)
    r1_vec, r2_vec = np.zeros((3,N)), np.zeros((3,N))

    r1_vec[0], r1_vec[1], r1_vec[2] = x1[:], y1[:], z1[:]
    r2_vec[0], r2_vec[1], r2_vec[2] = x2[:] ,y2[:], z2[:]

    return np.exp(-2*alpha*(r1 + r2)) / np.linalg.norm(r1_vec-r2_vec)

def MonteCarloNormal(N,lmbda):

    a=-lmbda
    b=-a

    x1,y1,z1,x2,y2,z2 = np.random.uniform(-lmbda,lmbda,(6,N))
    fx=function(x1,x2,y1,y2,z1,z2,N)

    crude_mc = np.sum(fx)
    integral=crude_mc*(b-a)/N
#    standard_dev = np.sqrt(np.sum(func_values - mean_value)**2)/(N)

#    total_mean_value = np.sum(mean_value)
#    total_standard_dev = np.sum(standard_dev)

    return integral#,standard_dev

integral=MonteCarloNormal(10000000,500)

print("Computed integral value: {}".format(integral))
#print("Standard Deviation: {}" .format(standard_deviation))
print("Actual integral value: {}".format(5*np.pi**2/16**2))
