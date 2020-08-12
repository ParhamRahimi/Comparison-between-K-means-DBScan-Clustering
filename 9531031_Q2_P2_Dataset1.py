import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import scipy.cluster.hierarchy as shc

variables = pd.read_csv('Dataset1.csv')
Y = variables[['X']]
X = variables[['Y']]

plt.figure(figsize=(10, 7))
plt.title("Dataset1 Dendograms")
dend = shc.dendrogram(shc.linkage(variables, method='ward'))