"""
Created on Sat Oct 15 16:07:25 2016
@author: rmisra
"""
import numpy
import matplotlib.pylab as py

# initializing values
numSamples = 10000000
numBits = 10
Z = 128
alpha = 0.2
B_i_0 = 0.5
B_i_1 = 0.5

numpy.random.seed(0);

# randomly sampling the bits 'numSamples' times according to the given probability distribution
B = [[numpy.random.choice(range(0,2), p = [B_i_0, B_i_1]) for x in range(numBits)] 
      for y in range(numSamples)]

# calculating f(B) for each of the samples
f_B = numpy.zeros((numSamples), dtype='int')
for i in range(0,numSamples):
    for j in range (0,numBits):
        f_B[i] = f_B[i] + (2**j)*B[i][j]
        
# calculating P(Z|B1,B2,....,BN) for each of the samples
P_Z = numpy.zeros((numSamples))
for i in range(0,numSamples):
    P_Z[i] = ((1-alpha)/(1+alpha))*(alpha**abs(Z - f_B[i]))

    
# calculating P(Bi = 1 | Z = 128) for i in {2,4,6,8,10} 
B_even_i_given_Z = numpy.zeros((5))
for j in range(0,5):
    sum_E = 0;          # to track total evidence across all the samples
    weighted_sum = 0;   # to track weighted likelihood of evidence
    
    # array to store the points to be plotted
    plt = numpy.zeros(int(numSamples/10));

    for i in range(0,numSamples):
        if(B[i][j*2+1] == 1):
            weighted_sum += P_Z[i]
        sum_E += P_Z[i]
        # storing the points to be plotted
        if((i+1)%10) == 0:
            plt[int(i/10)] = (weighted_sum/sum_E)   # inference till ith sample
        
    B_even_i_given_Z[j] = weighted_sum/sum_E
    
    # plotting begins 
    py.plot(range(0, len(plt)*10, 10), plt, 'r-')
    py.ylim(-0.05, 1.1)
    py.yticks(numpy.arange(0.0, 1.1, 0.1))
    py.tick_params(labelright = True)
    py.xlabel('Number of Samples')
    py.ylabel('Probability')
    py.title("P(B" + str(j*2+2) + "  = 1 | Z = 128)")
    py.savefig("B" + str(j*2+2) + ".pdf", bbox_inches='tight')
    py.show()

# printing the results
for j in range(0,int(numBits/2)):
    print("P(B" + str(j*2+2) + "  = 1 | Z = 128) = " + str(B_even_i_given_Z[j]));