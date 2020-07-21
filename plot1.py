import numpy as np
import matplotlib.pyplot as plt

name = 'proteins_rcmk_heatmap'

with open(name + '.txt', 'r') as f:
    n = int(f.readline().strip())
    data = np.array([list(map(int, line.strip().split())) for line in f.readlines()])

print(data)
data = data[:1000, :1000]

fig, ax = plt.subplots()
im = ax.imshow(data)
cbar = ax.figure.colorbar(im, ax=ax)

ax.set_title(name)
fig.tight_layout()

plt.savefig(name + 'part.pdf')

