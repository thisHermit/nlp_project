import site
import os

# Get the site-packages directory
site_packages_path = site.getsitepackages()[0]
print(f"Site-packages path: {site_packages_path}")

# Define the path to the file
file_path = os.path.join(site_packages_path, 'sophia', '__init__.py')

# Read the contents of the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Modify the first line
if lines:
    lines[0] = lines[0].replace(", sophiag", "")

# Write the modified contents back to the file
with open(file_path, 'w') as file:
    file.writelines(lines)

print(f"Modified the first line of {file_path}")
