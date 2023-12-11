#!/usr/bin/env python3
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

array = np.genfromtxt("array.csv")
bins = np.genfromtxt("bins.csv")
df = pd.DataFrame({'Values': array})

sns.set_theme()
sns.displot(data=df, bins=bins, x='Values')
sns.rugplot(data=df, x='Values', height=-0.02, clip_on=False, alpha=0.5, color='g')
mpl.pyplot.show()
