# OPTIMIZE

# INITIALIZE EMBEDDING VECTORS! 

from operator import add
import numpy as np
import pandas as  pd
import os

print(os.getcwd())

import time

start = time.time()
print("hello")
end = time.time()
print(end - start)

df = pd.read_csv('elements.csv')
for i in df.columns:
    print (i)

Element=df['Element'].tolist()
Symbol=df['Symbol'].tolist()

features = [Symbol, Element]  # a list of lists  # in this case the second index will attack inside the list, # first index is the feature # can have as many features as you want

p = 6
print(features[0][p], features[1][p])  

print(len(features))
print(len(features[0]))


numberoffeatures=2
numberofdatapoints = 10 #118  # H, He, Li,...
lengthofembeddingvector = 70

#initialize training hyperparameters
training_iterations = 20


import random




em = [[[0 for _ in range(lengthofembeddingvector)] for i in range(numberofdatapoints)] for i in range(numberoffeatures)]



for i in range(len(em)):
    for j in range(len(em[0])):
        for k in range(len(em[0][0])):
          
            em[i][j][k] = random.random()




# OPTIMIZE

def optimize_all_embeddings(em) -> list:
    
    

    
    
    

    
    def loss(em):
        datapoints = len(em[0])
    
     
     
        def distance(pair):
            distance = 0  # some default value 
            for k in range(lengthofembeddingvector):
                distance_per_element = pair[0][k]*pair[1][k]
                distance += distance_per_element
                
                distance = distance/lengthofembeddingvector  # normalize by number of elements
        
            return distance
             

        
     
        def distance_deprecated(pair):    # this is taking a point wise difference , this is not cosine similarity, its still a difference, so what does it mean
            distance = 0  # some default value 
            for k in range(lengthofembeddingvector):
                distance_per_element = pair[0][k] - pair[1][k]
                distance += distance_per_element
        
            return distance
        
        
        numerator = 0
        for i in range(datapoints):
            
            pair =   [em[0][i], em[1][i]]   # these pairs are aligned to be the same 
            
            d = distance(pair)
            numerator +=d
            
            denominator = 0  
            for j in range(datapoints):
                    if j==i:
                        continue
                    else:
                        pair =   [em[0][j], em[1][j]] # these pairs are different # negative marking 
                        d = distance(pair)
                        denominator +=d
                        
            loss = numerator / denominator
                        
        return loss
    
    def repeated_func(vector, em):
                        shift = [random.random() for i in range(lengthofembeddingvector)]
                        proposed_vector  = list( map(add, vector, shift) )  # this proposes 1 new vector
                        proposed_em = em
                        proposed_em[i][j] = proposed_vector
                        new_loss = loss(proposed_em)
                        return new_loss, proposed_em
            
    #track loss
    loss_track=[]
        
        
    proposed_em = em
    current_loss = 1000
    # START ITERATION 
    # Iteration to find optimal embeddings
    for i in range(training_iterations):   
        
            

            for i in range(len(em)):  # for every feature
            
                for j in range(len(em[0])):  # for every datapoint
                    
                    
                    loss_track.append(current_loss)
                    vector = em[i][j]
                    new_loss, proposed_em= repeated_func(vector, em)
                    
                    if np.absolute(new_loss) < np.absolute(current_loss):  # this never accepts higher energy solutions, so why is the loss increasing 
                        
                        # need a probability of acceptance 
                        
                        em = proposed_em # update the variable that stores embeddings 
                        current_loss=new_loss #isn't this redundant since current loss is calculated based on em 
                
                        
                    
    loss_track.pop(0)
    return em, loss_track

# RUN OPTIMIZATION and PLOT LOSS

import matplotlib.pyplot as plt



em, loss_track = optimize_all_embeddings(em)


# we want to iterate over every embedding vector.  



iteration = [i for i in range(len(loss_track))]

plt.plot(iteration, loss_track, '--co', label='line with marker')
#plt.scatter(iteration, loss)
plt.show()


plt.plot(iteration, loss_track, '--co', label='line with marker')
plt.ylim((-5,5))
plt.show()

