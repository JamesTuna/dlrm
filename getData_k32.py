#! /usr/bin/env python3
import torch,argparse
import numpy as np
from scipy.stats import entropy
import scipy,time,pickle
import matplotlib.pyplot as plt

'''This script extract 21 embedding matrices from
    1).dlrm_k.m
    2).linp_32_k.m'''

def effective_rank(X):
    U,S,VT = np.linalg.svd(X)
    S = S / np.linalg.norm(S,1)
    assert (np.all(S >= 0) and abs(S.sum()-1) < 1e-5), 'S not a probability distribution'
    H = entropy(S)
    return np.exp(H)

def fast_effective_rank(X):
    m,n = X.shape
    if m > n:
        square = X.T.dot(X)
    else:
        square = X.dot(X.T)
    U,S,VT = np.linalg.svd(square)
    S = np.sqrt(S)
    S = S / np.linalg.norm(S,1)
    assert (np.all(S >= 0) and abs(S.sum()-1) < 1e-5), 'S not a probability distribution'
    H = entropy(S)
    return np.exp(H)

parser = argparse.ArgumentParser(description='This script extract 21 embedding matrices from dlrm_k.m and linp_32_k.m')
parser.add_argument('--k',type=int,help='embedding dimension')
args = parser.parse_args()

K = args.k
assert K in (16,32), 'cannot handle K=%s'%(K)

sd_k = torch.load('../dlrm_'+str(K)+'.m')['state_dict']
sd_32_k = torch.load('../linp_32_'+str(K)+'_normal.m')['state_dict']

layer1_tables = [None for i in range(21)]
layer2_tables = [None for i in range(21)]

for name, value in sd_k.items():
    header,index,_ = name.split('.')
    if header == 'emb_l':
        print('1 layer, %s'%(str(name)))
        n, k = list(value.size())
        assert k == K, '1 layer embedding dimension doesn\'t match, specified %s but get %s'%(K,k)
        layer1_tables[int(index)] = torch.Tensor.cpu(value).numpy()

for name, value in sd_32_k.items():

    header,index,_ = str(name).split('.')

    if header == 'emb_l':
        print('2 layer, %s'%(str(name)))
        n, d = list(value.size())
        assert d == 32, str(name)+': inner dimension doesn\'t match, specified 32 but get %s'%(d)
        layer2_tables[int(index)] = (torch.Tensor.cpu(value)).numpy()

    elif header == 'emb_linp':
        print('2 layer, %s'%(str(name)))
        k, d = list(value.size())
        assert d == 32, str(name)+': inner dimension doesn\'t match, specified 32 but get %s'%(d)
        layer2 = (torch.Tensor.cpu(value).numpy()).T
        layer2_tables[int(index)] = layer2_tables[int(index)].dot(layer2)

layer1_efranks = np.zeros(21)
layer2_efranks = np.zeros(21)
for i in range(21):
    assert layer1_tables[i].shape == layer2_tables[i].shape,'shape error for i=%s'%i
    n, k = layer1_tables[i].shape

    efrank1 = fast_effective_rank(layer1_tables[i])
    efrank2 = fast_effective_rank(layer2_tables[i])
    layer1_efranks[i] = efrank1
    layer2_efranks[i] = efrank2
    print('table %02d shape (%6d x %6d)\nEffective Rank: 1-layer model %.4f 2-layer model %.4f'%(i,n,k,efrank1,efrank2))

print('save data')
#pickle.dump((layer1_tables,layer2_tables),open('data/dimension_'+str(K)+'_tables.pkl','wb'))
pickle.dump((layer1_efranks,layer2_efranks),open('data/inner_32_dimension_'+str(K)+'_tables.pkl','wb'))
print('save completed')
