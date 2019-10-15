print('program begins')
import numpy as np
import matplotlib.pyplot as plt



def MonteCarloNormal(N):
    alpha=2
    lmbda=1000
    """
    attempt to avoid divison by zero by excluding this value from our domain
    """
    l1=np.random.uniform(-lmbda,-0.001,(6,int(N/2)))
    l2=np.random.uniform(0.001,lmbda,(6,int(N/2)))
    random_array=np.concatenate((l1,l2),axis=1)
    x1,y1,z1,x2,y2,z2=random_array
    r1=np.sqrt(x1**2 + y1**2 + z1**2)
    r2=np.sqrt(x2**2 + y2**2 + z2**2)
    r1_vec, r2_vec = np.zeros((3,N)), np.zeros((3,N))
    r1_vec[0], r1_vec[1], r1_vec[2] = x1[:], y1[:], z1[:]
    r2_vec[0], r2_vec[1], r2_vec[2] = x2[:] ,y2[:], z2[:]
    func_values = np.exp(-2*alpha*(r1 + r2)) / (np.linalg.norm(r1_vec + r2_vec))
    print(func_values)
    print(func_values.shape)
    func_values = func_values
    print(func_values.shape)
    mean_value = np.mean(func_values)
    standard_dev = np.sqrt(np.sum(func_values - mean_value)**2)/(N)

    total_mean_value = np.sum(mean_value)
    total_standard_dev = np.sum(standard_dev)

    return total_mean_value,total_standard_dev

mean_value,standard_deviation=MonteCarloNormal(1000000)
print(mean_value)
print(standard_deviation)
