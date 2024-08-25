import pandas as pd
from sklearn.model_selection import train_test_split

# Data lcation is:
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

def load_imdb_data(data_file):
    """
    Loads the IMDB dataset from a CSV file, and processes it.

    Args:
        data_file (str): Path to the CSV file containing the IMDB dataset.

    Returns:
        DataFrame: A DataFrame containing the processed data.
    """

    df = pd.read_csv(data_file)
    df = df.reset_index().rename(columns={'index': 'id'})
    df = df.rename(columns={'review': 'sentence'})
    df['sentiment'] = df['sentiment'].apply(lambda x: "1" if x.lower() == "positive" else "0")
    
    return df[['id', 'sentence', 'sentiment']]

def split_and_save_data(df, train_file, dev_file, test_file, test_size=0.6, dev_size=0.1, random_state=42):
    """
    Splits the data into training, development (dev), and test sets and saves them to CSV files.

    Args:
        df (DataFrame): DataFrame.
        train_file (str): Path to save the training data.
        dev_file (str): Path to save the development data.
        test_file (str): Path to save the test data.
        test_size (float): Proportion of the dataset to include in the test set.
        dev_size (float): Proportion of the remaining dataset (after test split) to include in the dev set.
        random_state (int): Random seed for reproducibility.
    """

    train_dev_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, dev_df = train_test_split(train_dev_df, test_size=dev_size, random_state=random_state)
    
    train_df.to_csv(train_file, sep='\t', index=False)
    dev_df.to_csv(dev_file, sep='\t', index=False)
    test_df.to_csv(test_file, sep='\t', index=False)

if __name__ == "__main__":

    data_file = "/user/ahmed.assy/u11454/nlp_project-main/imdb-data/IMDB Dataset.csv"  # Update this path with the correct location
    train_file = "data/IMDB_train_data.csv"
    dev_file = "data/IMDB_dev_data.csv"
    test_file = "data/IMDB_test_data.csv"
    
    df = load_imdb_data(data_file)
    
    split_and_save_data(df, train_file, dev_file, test_file)
    
    print(f"Training data saved to {train_file}")
    print(f"Development data saved to {dev_file}")
    print(f"Test data saved to {test_file}")
