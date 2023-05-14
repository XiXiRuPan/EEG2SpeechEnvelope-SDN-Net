import numpy as np
import matplotlib.pyplot as plt

# Create data
data = np.random.randn(500, 64)

# Create figure with 64 subplots
fig, axs = plt.subplots(64, 1, figsize=(100, 50))

# Set different colors for each subplot
colors = plt.cm.rainbow(np.linspace(0, 1, 64))

# Plot each subplot with different color
for i in range(64):
    axs[i].plot(data[:, i], color=colors[i])

# Add title and labels
fig.suptitle('64 Subplots with Different Colors', fontsize=20)
fig.text(0.5, 0.04, 'X-axis', ha='center', fontsize=16)
fig.text(0.04, 0.5, 'Y-axis', va='center', rotation='vertical', fontsize=16)

#plt.show()
fname="eeg.png"
plt.savefig(fname, dpi=None) 
