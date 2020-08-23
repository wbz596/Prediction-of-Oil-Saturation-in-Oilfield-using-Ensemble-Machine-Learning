import pandas
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
names = ['PF-1C','PF-5','PF-11','PF-12','PF-14','PF-15D','IF-4','IF-5']
#os.chdir(os.getcwd()) #os.getcwd()get current path，os.chdir(...)change path...
#input data
#data = pandas.read_table('matrix.txt',sep='\s+')  
#correlation coefficient
#correlations = data.corr(method='spearman') 
#correction=abs(correlations)

#read .csv file to fetch matrix
data = np.genfromtxt("matrix 3 combine 1 8.21.csv", delimiter=",")
matrix = data[1:,1:9]

# plot correlation matrix 
fig, ax = plt.subplots(figsize = (10,5))
ax.set_xticklabels(matrix,rotation='horizontal')
sns.heatmap(matrix, cmap=plt.cm.Reds, linewidths=0.1, vmax=1, vmin=0 ,annot=False, square = True,fmt='.2g')

#heatmap parameters（correlation，color，interval）
#ticks = numpy.arange(0,16,1) #range0-16，step = 1 
#plt.xticks(np.arange(25),names) #x axis note
#plt.yticks(np.arange(25),names) #y axis note
#ax.set_xticks(ticks) #scale
#ax.set_yticks(ticks)
ax.set_xticklabels(names,fontsize = 10, horizontalalignment='center',rotation = 45) #lable of x axis 
ax.set_yticklabels(names, fontsize = 10, horizontalalignment='right',rotation = 360)
ax.tick_params(axis='both', which='both', length=0)#delete dash symbol in axis
#ax.set_title('Characteristic correlation')#title setting
#plt.savefig('cluster.tif',dpi=300)
plt.show()