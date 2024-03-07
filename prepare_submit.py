# Creates a zip file for submission.

import os
import sys
import zipfile

# Collect Python files
required_files = [p for p in os.listdir(".") if p.endswith(".py")]

# Collect predictions
for root, dirs, files in os.walk("predictions"):
    required_files += [os.path.join(root, file) for file in files]


def main():
    aid = "dnlp_final_project_submission"
    path = os.getcwd()
    with zipfile.ZipFile(f"{aid}.zip", "w") as zz:
        for file in required_files:
            zz.write(file, os.path.join(".", file))
    print(f"Submission zip file created: {aid}.zip")


if __name__ == "__main__":
    main()
