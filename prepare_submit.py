# Creates a zip file for submission.

import os
import zipfile

# Collect predictions
required_files = []
for root, dirs, files in os.walk("predictions"):
    required_files += [os.path.join(root, file) for file in files if not file.endswith(".gitkeep")]


def main():
    aid = "dnlp_final_project_submission"
    with zipfile.ZipFile(f"{aid}.zip", "w") as zz:
        for file in required_files:
            zz.write(file, os.path.join(".", file))
    print(f"Submission zip file created: {aid}.zip")


if __name__ == "__main__":
    main()
