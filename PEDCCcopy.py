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
#print('v',v)
# classical Gram-Schmidt orthogonalization
# A is a matrix (float numpy array) whose columns are linearly independent
#guistart=time.time()
#s=np.zeros((out_dim,out_dim))
#ss=np.zeros((out_dim,out_dim))

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
#guiend=time.time()
#print('zhengjia',guiend-guistart)
a=[1]
a1=[-1]
zero=np.zeros(latent_variable_dim-class_num+1)
zero=zero.tolist()
#print(zero)
a=a[:0]+zero+a[0:]
a1=a1[:0]+zero+a1[0:]
#a=np.array(a)
#a1=np.array(a1)
u=np.stack((a1,a))
u=u.tolist()
#print(type(a1),type(a),u)

#构建
#end=time.time()
#print(end-start,len(u))
#jianstart=time.time()
for i in range(class_num-2):
    #print(len(u))

    #print('u',u[len(u)-1])
    c=np.insert(u[len(u)-1],0,0)
    #c.insert(0,0)
 #   print('c',c)
#    print(len(u),type(u))
    for j in range(len(u)):
  #      print('j',type(u[j]))
        p=np.append(u[j],0).tolist()
   #     print('j', type(u[j]))
    #    print('b',b)
        s=len(u)+1
        u[j]=math.sqrt(s*(s-2))/(s-1)*np.array(p)-1/(s-1)*np.array(c)
     #   print(j,'u[j]',u[j])
    u.append(c)

u=np.array(u)
#print(u.shape)
#print('u',u.shape)
#print('v',v,np.linalg.det(v))
#print('b',np.dot(u[0,:],u[6,:].reshape((-1,1))))
# t('classical_gs(v): ', classical_gs(v))
g = np.dot(np.array(u), classical_gs(v))
g_T = classical_gs(v).transpose()

#print('shape',b.shape,np.array(u).shape)
#print('shimizizhengjiaohua(v)',b,np.linalg.det(b))
#print(g.shape)
#print('a',np.dot(g[1,:],g[4,:].reshape((-1,1))))
      #(u),np.array(u).shape)

jian = time.time()
#print('zui',jian-start)
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

## distance matrix
# result=np.zeros((len(b),len(b)))
# for a in range(len(b)):
#     for aa in range(a,len(b)):
#         result[a][aa]=np.linalg.norm((b[a]-b[aa]))
#         result[aa][a] = result[a][aa]
fff=open(PEDCC_ui,'wb')
map={}
for i in range(len(b)):
    map[i]=torch.from_numpy(np.array([b[i]]))
pickle.dump(map,fff)
fff.close()




