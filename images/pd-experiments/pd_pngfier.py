import pandas as pd
import matplotlib.pyplot as plt
import sys

# Read the CSV file
df = pd.read_csv(sys.argv[1])

# Create a single figure for Dev and Train scores
fig, ax = plt.subplots(figsize=(12, 6))

# Plot Dev and Train scores
ax.plot(df['Epoch'], df['Train Acc'], marker='o', linestyle='-', label='Train Acc')
ax.plot(df['Epoch'], df['Dev Accuracy'], marker='s', linestyle='-', label='Dev Accuracy')

# Customize the plot
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Epoch')
ax.grid(True)
ax.legend()
ax.set_ylim(bottom=0.6)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(f'{sys.argv[1]}_scores_vs_epoch.png', dpi=300, bbox_inches='tight')

# Close the plot to free up memory
plt.close(fig)

print("Plot has been generated and saved.")