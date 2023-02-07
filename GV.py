import numpy as np
import pandas as pd



#you dont need the permutation matrix in cuda just do element wise vector to row

#random Graph generator
size = np.random.randint(low=90,high=100)
Graph = np.zeros(shape=(size,size))
tenth = 1
Side = np.zeros(shape=(size))
print("this might take a few minutes")
for i in range(size):
    for j in range(i+1,size):
        edge = int(round(np.random.rand()+0.1))
        Graph[i][j] = edge
        Graph[j][i] = edge
    if(i>size*tenth/10):
        print("T: ",tenth)
        tenth = tenth +1
    
        

    if np.random.rand() < 0.5:
        Side[i] = -1
    else:
        Side[i] = 1

    Graph[i][i] = 0
pd.DataFrame(Graph.astype(np.int32)).to_csv(r"./Graph.txt",sep=",",index=False,header=False)
pd.DataFrame(Side.astype(np.int32)).to_csv(r"./Side.txt",sep=",",index=False,header=False)
print("now run nvcc -lcublas -lcurand kernel.cu")
# print(Graph)



# # row * abs(Side + IverseSide)/2 CUT SIZE
# for i in range(size):
#     # res = np.dot(Graph[i],(np.abs((Side + -Side[i])/2)))
#     res = np.dot(Graph[i],-Side*Side[i])
#     print(res)
# print("____________________")
# # Balance of each side
# leftB, rightB = np.sum((np.abs((Side - 1)/2))),np.sum(np.abs((Side + 1)/2))
# print(leftB,rightB)

# print("____________________")


# for i in range(size):
#     for j in range(size):
#         Graph[i][j] = Side[i] * Graph[i][j]

# for i in range(size):
#     for j in range(size):
#         Graph[j][i] = Side[i] * Graph[j][i]
# print("GG",Graph)
# print(-np.sum(((Graph - 1)/2).astype(np.int8))/2)
# print("SUM",Side)
#a vector of blocked cels
# we hav