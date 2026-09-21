import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


fname = r'data\processed\per_pixel_heights.csv'
df = pd.read_csv(fname, index_col = 0)

fig,ax = plt.subplots(1,1)
fig.set_size_inches(8,8)
ax.set_aspect('equal')
ax.scatter(df['median_height'], df['weighted_mean_height'], color = 'black', alpha = 0.05, s = 0.2)
ax.plot([-50,50],[-50,50],color = 'red',linestyle = '--',linewidth = 1,alpha = 0.5)
ax.set_xlabel('Median determined height')
ax.set_ylabel('Weighted mean determined height')
ax.set_xlim(-12,0)
ax.set_ylim(-12,0)
plt.tight_layout()
plt.show()

print(np.nanstd(df['median_height']-df['weighted_mean_height']))
print(df['median_height'].corr(df['weighted_mean_height']))