import pandas as pd
import matplotlib.pyplot as plt
import sys

# Read the CSV file
df = pd.read_csv(sys.argv[1])

# Create two figures: one for accuracies and one for Matthews coefficients
fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
fig_mcc, ax_mcc = plt.subplots(figsize=(12, 6))

# Plot accuracies
ax_acc.plot(df['Epoch'], df['Validation Accuracy'], marker='o', linestyle='-', label='Validation Accuracy')
ax_acc.plot(df['Epoch'], df['Total Train Accuracy'], marker='s', linestyle='-', label='Train Accuracy')

# Plot Matthews coefficients
ax_mcc.plot(df['Epoch'], df['Validation Matthews Coefficient'], marker='o', linestyle='-', label='Validation Matthews Coefficient')
ax_mcc.plot(df['Epoch'], df['Total Train Matthews Coefficient'], marker='s', linestyle='-', label='Train Matthews Coefficient')

# Customize accuracy plot
ax_acc.set_xlabel('Epoch')
ax_acc.set_ylabel('Accuracy')
ax_acc.set_title('Accuracy vs Epoch')
ax_acc.grid(True)
ax_acc.legend()
ax_acc.set_ylim(bottom=0.7)  # Set y-axis to start at 0.8

# Customize Matthews coefficient plot
ax_mcc.set_xlabel('Epoch')
ax_mcc.set_ylabel('Matthews Coefficient')
ax_mcc.set_title('Matthews Coefficient vs Epoch')
ax_mcc.grid(True)
ax_mcc.legend()

# Adjust layout and save the plots
plt.figure(fig_acc.number)
plt.tight_layout()
plt.savefig(f'{sys.argv[1]}_accuracies_vs_epoch.png', dpi=300, bbox_inches='tight')

plt.figure(fig_mcc.number)
plt.tight_layout()
plt.savefig(f'{sys.argv[1]}_matthews_coefficients_vs_epoch.png', dpi=300, bbox_inches='tight')

# Close the plots to free up memory
plt.close(fig_acc)
plt.close(fig_mcc)

print("Merged plots have been generated and saved.")
