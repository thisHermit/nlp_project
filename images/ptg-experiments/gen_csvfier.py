import re
import csv
import sys

# Function to extract metrics from the given output file
def extract_metrics(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Regular expressions to extract the required metrics
    train_score_pattern = r"Train score: ([0-9.]+)"
    dev_score_pattern = r"Dev score: ([0-9.]+)"
    
    train_scores = [float(x) for x in re.findall(train_score_pattern, data)]
    dev_scores = [float(x) for x in re.findall(dev_score_pattern, data)]
    
    return train_scores, dev_scores

# Function to save metrics to a CSV file
def save_metrics_to_csv(train_scores, dev_scores, csv_file_path):
    # Creating a list of dictionaries for each epoch's data
    data = []
    for epoch in range(len(train_scores)):
        data.append({
            'Epoch': epoch + 1,
            'Train Score': train_scores[epoch],
            'Dev Score': dev_scores[epoch]
        })
    
    # Writing the data to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Epoch', 'Train Score', 'Dev Score'])
        writer.writeheader()
        writer.writerows(data)
    print(f"Metrics saved to {csv_file_path}")

# Main function
def main():
    file_path = sys.argv[1]  # Replace with your file path
    csv_file_path = f'{sys.argv[1]}-metrics.csv'  # Replace with your desired CSV file path
    train_scores, dev_scores = extract_metrics(file_path)
    save_metrics_to_csv(train_scores, dev_scores, csv_file_path)

if __name__ == "__main__":
    main()