#! /usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pickle,argparse

parser = argparse.ArgumentParser(description='This script compare effective rank across different inner dimension')
parser.add_argument('--k',type=int,help='embedding dimension')
args = parser.parse_args()
K = args.k

inner_8 = pickle.load(open('data/inner_8_dimension_%s_tables.pkl'%K,'rb'),encoding='latin1')
inner_16 = pickle.load(open('data/inner_16_dimension_%s_tables.pkl'%K,'rb'),encoding='latin1')
#inner_32 = pickle.load(open('data/inner_32_dimension_%s_tables.pkl'%K,'rb'),encoding='latin1')
inner_64 = pickle.load(open('data/dimension_%s_tables.pkl'%K,'rb'),encoding='latin1')


efrank_8 = inner_8[1]
efrank_16 = inner_16[1]
#efrank_32 = inner_32[1]
efrank_64 = inner_64[1]
efrank_1layer = inner_8[0]
print(np.abs(inner_8[0]-inner_16[0]).sum())
print(np.abs(inner_16[0]-inner_64[0]).sum())

Xs = range(21)
plt.figure(figsize=(12,8))
l1 = plt.scatter(Xs,efrank_8,marker='o',s=120,alpha=0.7)
l2 = plt.scatter(Xs,efrank_16,marker='o',s=120,alpha=0.7)
l3 = plt.scatter(Xs,efrank_64,marker='o',s=120,alpha=0.7)

plt.legend([l1,l2,l3],['inner 8','inner 16','inner 64'],loc='best')
plt.title('Effective Ranks of different k when embedding dim=%s'%K)
plt.xlabel('Sparse Feature Index')
plt.ylabel('Effective Rank')
plt.xticks(Xs)
plt.grid(linestyle='--')
plt.savefig('plots/k_compare_dim=%s.pdf'%K)
plt.show()
