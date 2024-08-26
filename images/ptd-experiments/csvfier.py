import re
import csv
import sys

# Function to extract metrics from the given output file
def extract_metrics(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Regular expressions to extract the required metrics
    val_acc_pattern = r"Validation:\n Accuracy: ([0-9.]+)"
    val_mcc_pattern = r"Validation:\n Accuracy: [0-9.]+ Matthews Coefficient: ([0-9.]+)"
    train_acc_pattern = r"Total Train:\n Accuracy: ([0-9.]+)"
    train_mcc_pattern = r"Total Train:\n Accuracy: [0-9.]+ Matthews Coefficient: ([0-9.]+)"
    
    val_acc = [float(x) for x in re.findall(val_acc_pattern, data)]
    val_mcc = [float(x) for x in re.findall(val_mcc_pattern, data)]
    train_acc = [float(x) for x in re.findall(train_acc_pattern, data)]
    train_mcc = [float(x) for x in re.findall(train_mcc_pattern, data)]
    
    return val_acc, val_mcc, train_acc, train_mcc

# Function to save metrics to a CSV file
def save_metrics_to_csv(val_acc, val_mcc, train_acc, train_mcc, csv_file_path):
    # Creating a list of dictionaries for each epoch's data
    data = []
    for epoch in range(len(val_acc)):
        data.append({
            'Epoch': epoch + 1,
            'Validation Accuracy': val_acc[epoch],
            'Validation Matthews Coefficient': val_mcc[epoch],
            'Total Train Accuracy': train_acc[epoch],
            'Total Train Matthews Coefficient': train_mcc[epoch]
        })
    
    # Writing the data to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Epoch', 'Validation Accuracy', 'Validation Matthews Coefficient', 'Total Train Accuracy', 'Total Train Matthews Coefficient'])
        writer.writeheader()
        writer.writerows(data)
    print(f"Metrics saved to {csv_file_path}")

# Main function
def main():
    file_path = sys.argv[1]  # Replace with your file path
    csv_file_path = f'{sys.argv[1]}-metrics.csv'  # Replace with your desired CSV file path
    val_acc, val_mcc, train_acc, train_mcc = extract_metrics(file_path)
    save_metrics_to_csv(val_acc, val_mcc, train_acc, train_mcc, csv_file_path)

if __name__ == "__main__":
    main()
