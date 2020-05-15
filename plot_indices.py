"""
Manuscript:
Modeling the transmission dynamics and the impact of the control
interventions for the COVID-19 epidemic outbreak

Authors:
Fernando Saldana, Hugo Flores-Arguedas, Jose Ariel Camacho-Gutiérrez, and Ignacio
Barradas

Date:
May, 2020

Before runing this script, the script Sens_covi.py must be run to generate the information.

"""



import numpy as np
import pylab as plt
import seaborn as sns
sns.set(style="ticks")
plt.rc('font', size=7)
sns.set()
sns.set_color_codes()

N=11

Index1 = np.load('indices/Indices1.npy')
Index1_conf = np.load('indices/Indices1_conf.npy')
Index2 = np.load('indices/Indices2.npy')
IndexT = np.load('indices/IndicesT.npy')
IndexT_conf = np.load('indices/IndicesT_conf.npy')

# Perform analysis
S1 = Index1
ST = IndexT
S2 = Index2

# width of the bars
barWidth = 0.3

# Choose the height of the blue bars
bars1 = S1

# Choose the height of the cyan bars
bars2 = ST

# Choose the height of the error bars (bars1)
yer1 = Index1_conf

# Choose the height of the error bars (bars2)
yer2 = IndexT_conf

# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(5,3))

# Create blue bars
plt.bar(r1, bars1, width = barWidth, color='b', yerr=yer1,capsize=7, label='S1')

# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'r', yerr=yer2, capsize=7, label='ST')

# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], ['$\\beta_A$', '$\\beta_I$', '$\\beta_V$','$c_1$','$c_2$'], fontsize=14)

plt.ylabel('Index', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('1order_Indices.pdf')
# Show graphic
#plt.close()
