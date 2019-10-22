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

def main(N):
    """Distributing the work over the processors """
    N_list=np.full(int(np.sqrt(N)),int(np.sqrt(N)))

    number_of_processors=mp.cpu_count()

    pool = mp.Pool(number_of_processors)

    result= np.array(pool.map( MonteCarlo, N_list ))
    result_int,result_var=result.T
#    print("Integral calculated using parallelization: {:.3f}".format(np.mean(result_int)))
#    print("Variance calculated using parallelization: {}".format(np.mean(result_var)))
    return np.mean(result_int), np.mean(result_var)

integral_list=[]
variance_list=[]
parallell_int_list=[]
parallell_var_list=[]
normal_time_list=[]
parallell_time_list=[]
N_iteration_list = np.array([1_00,1_00_0,1_00_00,1_00_00_0,1_00_00_00,1_00_00_00_0])


iterations=1
for N in N_iteration_list:

    start=time.time()
    integral,variance=MonteCarlo(N)
    end=time.time()
    print("time not parallell {}".format(end-start))

    start2=time.time()
    parallell_int, parallell_var = main(N)
    end2=time.time()
    print("time parallell {}".format(end2-start2))

    integral_list.append(integral)
    variance_list.append(variance)
    normal_time_list.append(end-start)

    parallell_int_list.append(parallell_int)
    parallell_var_list.append(parallell_var)
    parallell_time_list.append(end2-start2)

    print("Iterations: {}".format(iterations))
    print("N = {}".format(N))

    print("Integral value: {}".format(parallell_int))
    iterations +=1

"""
plt.plot(N_iteration_list[1:], integral_list[1:])
plt.scatter(N_iteration_list, integral_list)
plt.xlabel("Number of iterations N",fontsize=16)
plt.ylabel("Integral values",fontsize=16)
plt.title("Integral values for Exponential distribution",fontsize=16)
plt.tight_layout()
plt.savefig("exp_int.pdf")
plt.close()


plt.plot(N_iteration_list, parallell_int_list)
plt.scatter(N_iteration_list, parallell_int_list)
plt.xlabel("Number of iterations N",fontsize=16)
plt.ylabel("Integral values",fontsize=16)
plt.title("Integral values for Exponential distribution, parallell",fontsize=16)
plt.tight_layout()
plt.savefig("exp_int_parallell.pdf")
plt.close()

plt.plot(N_iteration_list, normal_time_list)
plt.scatter(N_iteration_list, normal_time_list)
plt.xlabel("Number of iterations N",fontsize=16)
plt.ylabel("Time in seconds",fontsize=16)
plt.title("Time to compute integral",fontsize=16)
plt.tight_layout()
plt.savefig("exp_time.pdf")
plt.close()


plt.plot(N_iteration_list, parallell_time_list)
plt.scatter(N_iteration_list, parallell_time_list)
plt.xlabel("Number of iterations N",fontsize=16)
plt.ylabel("Time in seconds",fontsize=16)
plt.title("Time to compute integral, parallell",fontsize=16)
plt.tight_layout()
plt.savefig("parallell_time.pdf")
plt.close()



plt.plot(N_iteration_list, variance_list)
plt.scatter(N_iteration_list, variance_list)
plt.xlabel("Number of iterations N",fontsize=16)
plt.ylabel("Variance",fontsize=16)
plt.title("Variance as a function of N",fontsize=16)
plt.tight_layout()
plt.savefig("variance.pdf")
plt.close()



plt.plot(N_iteration_list, parallell_var_list)
plt.scatter(N_iteration_list, parallell_var_list)
plt.xlabel("Number of iterations N",fontsize=16)
plt.ylabel("Variance",fontsize=16)
plt.title("Variance as a function of N, parallell",fontsize=16)
plt.tight_layout()
plt.savefig("parallell_variance.pdf")
plt.close()
"""
