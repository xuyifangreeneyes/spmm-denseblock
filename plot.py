import numpy as np
import matplotlib.pyplot as plt

name = 'ddi_bfs_heatmap'

with open(name + '.txt', 'r') as f:
    n = int(f.readline().strip())
    # lines = f.readlines()
    # print(lines[0])
    # print(lines[-1])
    # print(len(lines))
    # print(len(f.readlines()))
    data = np.array([list(map(int, line.strip().split())) for line in f.readlines()])

print(data)

fig, ax = plt.subplots()
im = ax.imshow(data)
cbar = ax.figure.colorbar(im, ax=ax)

ax.set_title(name)
fig.tight_layout()

plt.savefig(name + '.pdf')

