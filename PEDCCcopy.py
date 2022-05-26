import torch
import math
import numpy as np
import time
import pickle
import os
start=time.time()
class_num = 30##任意k个点
a = [1]
latent_variable_dim = 256##n维欧式空间
PEDCC_root=r"./center_pedcc/"
PEDCC_ui=os.path.join(PEDCC_root,str(class_num)+"_"+str(latent_variable_dim)+"_s.pkl")

if not os.path.isdir(PEDCC_root):
    os.makedirs(PEDCC_root)


v = np.random.rand(latent_variable_dim, latent_variable_dim)  ##随机生成n乘n的方阵


def classical_gs(A):
    dim = A.shape
    #print(dim,type(dim))
    Q = np.zeros(dim) # initialize Q
    R = np.zeros((dim[1], dim[1]))  # initialize R
    for j in range(dim[1]):
        y = np.copy(A[:, j])
        for i in range(j):
            R[i, j] = np.matmul(np.transpose(Q[:, i]), A[:, j])
            y -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(y)
        Q[:, j] = y / R[j, j]
    return Q
a=[1]
a1=[-1]
zero=np.zeros(latent_variable_dim-class_num+1)
zero=zero.tolist()
a=a[:0]+zero+a[0:]
a1=a1[:0]+zero+a1[0:]
u=np.stack((a1,a))
u=u.tolist()

#构建
for i in range(class_num-2):
    c=np.insert(u[len(u)-1],0,0)
    for j in range(len(u)):
        p=np.append(u[j],0).tolist()
        s=len(u)+1
        u[j]=math.sqrt(s*(s-2))/(s-1)*np.array(p)-1/(s-1)*np.array(c)
    u.append(c)

u=np.array(u)
g = np.dot(np.array(u), classical_gs(v))
g_T = classical_gs(v).transpose()

jian = time.time()
G=np.zeros((class_num,class_num))
for i in range(class_num):
    for j in range(class_num):
         G[i,j] =  np.dot(g[i,:],g[j,:]) / (np.linalg.norm(g[i,:]) * np.linalg.norm(g[j,:]))
print(G)
f=open('./tmp.pkl', 'wb')
pickle.dump(g,f)
f.close()

ff=open("./tmp.pkl", 'rb')
b=pickle.load(ff)
ff.close()

os.remove("./tmp.pkl")

fff=open(PEDCC_ui,'wb')
map={}
for i in range(len(b)):
    map[i]=torch.from_numpy(np.array([b[i]]))
pickle.dump(map,fff)
fff.close()




