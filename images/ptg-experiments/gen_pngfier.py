import pandas as pd
import matplotlib.pyplot as plt
import sys

# Read the CSV file
df = pd.read_csv(sys.argv[1])

# Create a single figure for Dev and Train scores
fig, ax = plt.subplots(figsize=(12, 6))

# Plot Dev and Train scores
ax.plot(df['Epoch'], df['Dev Score'], marker='o', linestyle='-', label='Dev Score')
ax.plot(df['Epoch'], df['Train Score'], marker='s', linestyle='-', label='Train Score')

# Customize the plot
ax.set_xlabel('Epoch')
ax.set_ylabel('Score')
ax.set_title('Dev and Train Scores vs Epoch')
ax.grid(True)
ax.legend()

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(f'{sys.argv[1]}_scores_vs_epoch.png', dpi=300, bbox_inches='tight')

# Close the plot to free up memory
plt.close(fig)

print("Plot has been generated and saved.")