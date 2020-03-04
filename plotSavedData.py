#! /usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pickle

rank_4 = pickle.load(open('data/dimension_4_tables.pkl','rb'),encoding='latin1')
rank_8 = pickle.load(open('data/dimension_8_tables.pkl','rb'),encoding='latin1')
rank_16 = pickle.load(open('data/dimension_16_tables.pkl','rb'),encoding='latin1')
rank_32 = pickle.load(open('data/dimension_32_tables.pkl','rb'),encoding='latin1')
rank_64 = pickle.load(open('data/dimension_64_tables.pkl','rb'),encoding='latin1')

rank_4_avg = (rank_4[0].mean(),rank_4[1].mean())
rank_8_avg = (rank_8[0].mean(),rank_8[1].mean())
rank_16_avg = (rank_16[0].mean(),rank_16[1].mean())
rank_32_avg = (rank_32[0].mean(),rank_32[1].mean())
rank_64_avg = (rank_64[0].mean(),rank_64[1].mean())

ratio_4 = rank_4[1]/rank_4[0]
ratio_8 = rank_8[1]/rank_8[0]
ratio_16 = rank_16[1]/rank_16[0]
ratio_32 = rank_32[1]/rank_32[0]
ratio_64 = rank_64[1]/rank_64[0]

for i in range(5):
    efrank1,efrank2 = (rank_4,rank_8,rank_16,rank_32,rank_64)[i]
    K = (4,8,16,32,64)[i]
    Xs = [i for i in range(21)]
    print(efrank1)
    plt.figure(figsize=(12,8))
    l1 = plt.scatter(Xs,efrank1,color='blue',alpha=0.7,s=100)
    l2 = plt.scatter(Xs,efrank2,color='red',alpha=0.7,s=100)
    plt.legend([l1,l2],['single-layer k=%s'%K,'2-layer d=%s k=%s'%(64,K)],loc='best')
    plt.title('Effective Ranks of Embedding Matrices Learned by 2 models, averaged ratio: %.4f'%(efrank1.mean()/efrank2.mean()))
    plt.xlabel('Sparse Feature Index')
    plt.ylabel('Effective Rank')
    plt.xticks(Xs)
    plt.grid(linestyle='--')
    plt.savefig('plots/(%s %s)rank_plot.pdf'%(64,K))
    plt.show()


avg_rank1 = [i[0] for i in (rank_4_avg,rank_8_avg,rank_16_avg,rank_32_avg,rank_64_avg)]
avg_rank2 = [i[1] for i in (rank_4_avg,rank_8_avg,rank_16_avg,rank_32_avg,rank_64_avg)]
Ks = (4,8,16,32,64)
plt.figure(figsize=(12,8))
l1 = plt.scatter(Ks,avg_rank1,color='blue',marker='o',s=120,alpha=0.7)
l2 = plt.scatter(Ks,avg_rank2,color='red',marker='o',s=120,alpha=0.7)
l1, = plt.plot(Ks,avg_rank1,color='blue')
l2, = plt.plot(Ks,avg_rank2,color='red')
plt.legend([l1,l2],['single-layer','2-layer d=%s'%(64)],loc='upper left')
plt.title('Average Effective Ranks of Embedding Matrices Learned by 2 models')
plt.xlabel('Dimension k')
plt.ylabel('Average Effective Rank')
plt.xticks(Ks)
plt.grid(linestyle='--')
plt.savefig('plots/avg_rank_plot.pdf')
plt.show()


for i in range(5):
    ratios = (ratio_4,ratio_8,ratio_16,ratio_32,ratio_64)[i]
    K = (4,8,16,32,64)[i]
    Xs = [i for i in range(21)]
    print(efrank1)
    plt.figure(figsize=(12,8))
    l1 = plt.scatter(Xs,ratios,color='blue',alpha=0.7,s=100)
    plt.title('Ratio of Effective Ranks with Embedding Dimension %d'%(K))
    plt.xlabel('Sparse Feature Index')
    plt.ylabel('Ratio of Effective Rank(2-layer/1-layer)')
    plt.xticks(Xs)
    plt.grid(linestyle='--')
    plt.savefig('plots/(%s %s)ratio_plot.pdf'%(64,K))
    plt.show()


avg_rank1 = [i.mean() for i in (ratio_4,ratio_8,ratio_16,ratio_32,ratio_64)]
Ks = (4,8,16,32,64)
plt.figure(figsize=(12,8))
l1 = plt.scatter(Ks,avg_rank1,color='blue',marker='o',s=120,alpha=0.7)
l1, = plt.plot(Ks,avg_rank1,color='blue')
plt.title('Average Ratio of Effective Ranks')
plt.xlabel('Dimension k')
plt.ylabel('Average Ratio of Effective Rank(2-layer/1-layer)')
plt.xticks(Ks)
plt.grid(linestyle='--')
plt.savefig('plots/avg_ratio_plot.pdf')
plt.show()
