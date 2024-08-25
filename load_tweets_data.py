import os
import requests
import pandas as pd


def download_file(url, filename):
    """
    Download a file from a specified URL and save it to a local file.

    Args:
    - url (str): The URL to download the file from.
    - filename (str): The local file path where the downloaded file will be saved.

    Raises:
    - requests.HTTPError: If the request to download the file fails. (incase of any error)
    """
    response = requests.get(url)
    response.raise_for_status()  
    with open(filename, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded {filename}")

def process_dataset(file_path):
    """
    Process the data.

    Args:
    - file_path (str): The path to the CSV file to be processed.

    Returns:
    - pd.DataFrame: The processed DataFrame.
    """

    df = pd.read_csv(file_path)
    df['sentiment'] = df['label'].map(label_to_class)
    df['sentiment'] = (df['sentiment']).astype(str)
    df['id'] = df.index
    df = df[['id', 'tweet', 'sentiment']]
    df.rename(columns={'tweet': 'sentence'}, inplace=True)
    
    return df

if __name__ == "__main__":

    # URLs for the datasets in GitHub
    urls = {
        "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/train_preprocess.csv",
        "dev": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/valid_preprocess.csv",
        "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/test_preprocess.csv"
    }

    # Download the files and save them.
    for key, url in urls.items():
        download_file(url, f"data/tweets_{key}.csv")

    # Mapping from labels to class (sentiment).
    label_to_class = {
        'anger': 0,
        'fear': 1,
        'happy': 2,
        'love': 3,
        'sadness': 4
    }

    train_df = process_dataset("data/tweets_train.csv")
    dev_df = process_dataset("data/tweets_dev.csv")
    test_df = process_dataset("data/tweets_test.csv")

    train_df.to_csv("data/processed_tweets_train.csv", sep='\t', index=False)
    dev_df.to_csv("data/processed_tweets_dev.csv", sep='\t', index=False)
    test_df.to_csv("data/processed_tweets_test.csv", sep='\t', index=False)

    print("Datasets have been downloaded, processed, and saved successfully in the 'data' folder.")
