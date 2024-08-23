import argparse
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from sklearn.metrics import matthews_corrcoef
from optimizer import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from datasets import (
    SentencePairDataset,
    preprocess_string
)
import torch.nn.functional as F
from multitask_classifier import select_subset
import csv

TQDM_DISABLE = False

class EarlyStopping:
    "Function to stop the training early, if the validation loss doesn't improve after a predefined patience."
    def __init__(self, checkpoint_path, patience=5, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation Loss/Score Decreased/Increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_checkpoint_path)
        self.val_loss_min = val_loss


class BartWithClassifier(nn.Module):
    def __init__(self, latent_dims=7, num_labels=7):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=True, add_cross_attention=True)

        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        self.paraphrase_classifier = nn.Linear(self.bart.config.hidden_size * 2, 1)

    def embedding(self, input_ids, attention_mask=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        return cls_output

    def forward(self, input_ids, attention_mask=None):
        # use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        
        out = self.classifier(cls_output)

        return out
    
    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: Quora
        """
        ### TODO
        # Get embeddings for both sentences
        embedding_1 = self.embedding(input_ids_1, attention_mask_1)
        embedding_2 = self.embedding(input_ids_2, attention_mask_2)

        # Concatenate the embeddings
        combined_embedding = torch.cat((embedding_1, embedding_2), dim=1)

        # Pass through the paraphrase classifier
        logit = self.paraphrase_classifier(combined_embedding)

        return logit.squeeze(-1)


def transform_data(dataset, args, max_length=512):
    """
    dataset: pd.DataFrame

    Turn the data to the format you want to use.

    1. Extract the sentences from the dataset. We recommend using the already split
    sentences in the dataset.
    2. Use the AutoTokenizer from_pretrained to tokenize the sentences and obtain the
    input_ids and attention_mask.
    3. Currently, the labels are in the form of [2, 5, 6, 0, 0, 0, 0]. This means that
    the sentence pair is of type 2, 5, and 6. Turn this into a binary form, where the
    label becomes [0, 1, 0, 0, 1, 1, 0]. Be careful that the test-student.csv does not
    have the paraphrase_types column. You should return a DataLoader without the labels.
    4. Use the input_ids, attention_mask, and binary labels to create a TensorDataset.
    Return a DataLoader with the TensorDataset. You can choose a batch size of your
    choice.
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    
    tokenized = tokenizer(
        text=dataset['sentence1'].tolist(), 
        text_target=dataset['sentence2'].tolist(), 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )
    
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    
    if 'paraphrase_types' in dataset.columns:
        labels = dataset['paraphrase_types'].apply(eval).tolist()
        if args.multi_label:
            lbc1 = dataset['sentence1_segment_location'].apply(eval).tolist()
            lbc2 = dataset['sentence2_segment_location'].apply(eval).tolist()
            joined_lbcs = [l1 + l2 for l1, l2 in zip(lbc1, lbc2)]
            binary_labels = []
            for label in joined_lbcs:
                counter = Counter(label)
                binary_labels.append([counter[i + 1] // min(counter.values()) for i in range(7)])
        else:
            binary_labels = [[1 if (i + 1) in label else 0 for i in range(7)] for label in labels]
        labels_tensor = torch.tensor(binary_labels, dtype=torch.float32)
        
        data = TensorDataset(input_ids, attention_mask, labels_tensor)
    else:
        data = TensorDataset(input_ids, attention_mask)
    
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True if 'paraphrase_types' in dataset.columns else False)
    
    return dataloader


def train_model(model, train_data, simul_dataloader, dev_data, device, args, early_stopping=None):
    """
    Train the model. You can use any training loop you want. We recommend starting with
    AdamW as your optimizer. You can take a look at the SST training loop for reference.
    Think about your loss function and the number of epochs you want to train for.
    You can also use the evaluate_model function to evaluate the
    model on the dev set. Print the training loss, training accuracy, and dev accuracy at
    the end of each epoch.

    Return the trained model.
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss() if args.multi_label else nn.BCEWithLogitsLoss() # since the target labels are binary
    
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(train_data, disable=TQDM_DISABLE):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted_labels = (outputs > 0.5).float()
            correct_predictions += (predicted_labels == labels).float().sum().item()
            total_predictions += labels.numel()
        
        for batch in tqdm(
            simul_dataloader, desc=f"train-qqp-{epoch+1:02}", disable=TQDM_DISABLE
        ):
            b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                batch["token_ids_1"],
                batch["attention_mask_1"],
                batch["token_ids_2"],
                batch["attention_mask_2"],
                batch["labels"],
            )

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
            loss = F.binary_cross_entropy_with_logits(logits, b_labels.float())
            loss.backward()
            optimizer.step()        

        avg_loss = total_loss / len(train_data)
        train_accuracy = correct_predictions / total_predictions
        dev_accuracy, dev_matthews_coefficient = evaluate_model(model, dev_data, device)
        total_train_accuracy, total_train_matthews_coefficient = evaluate_model(model, train_data, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training step:\nLoss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Validation:\n Accuracy: {dev_accuracy} Matthews Coefficient: {dev_matthews_coefficient:.4f}")
        print(f"Total Train:\n Accuracy: {total_train_accuracy} Matthews Coefficient: {total_train_matthews_coefficient:.4f}")
        print()
        if early_stopping is not None:
            early_stopping(-dev_matthews_coefficient, model)
            if early_stopping.early_stop:
                print("Early Stopping...")
                break

    return model


def test_model(model, test_data, test_ids, device):
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    The 'Predicted_Paraphrase_Types' column should contain the binary array of your model predictions.
    Return this dataframe.
    """
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_data, disable=TQDM_DISABLE):
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_labels = (outputs > 0.5).int()
            all_predictions.extend(predicted_labels.cpu().numpy().tolist())

    results_df = pd.DataFrame({
        'id': test_ids,
        'Predicted_Paraphrase_Types': [str(pred) for pred in all_predictions]
    })

    return results_df


def evaluate_model(model, test_data, device):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the seven paraphrase types the model has predicted correctly for each data point.
    So, if the models prediction is [1,1,0,0,1,1,0] and the true label is [0,0,0,0,1,1,0], this predicition
    has an accuracy of 5/7, i.e. 71.4% .
    """
    all_pred = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_labels = (outputs > 0.5).int()

            all_pred.append(predicted_labels)
            all_labels.append(labels)

    all_predictions = torch.cat(all_pred, dim=0)
    all_true_labels = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels.cpu().numpy()
    predicted_labels_np = all_predictions.cpu().numpy()

    # Compute the accuracy for each label
    accuracies = []
    matthews_coefficients = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(
            true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx]
        )
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

        #compute Matthwes Correlation Coefficient for each paraphrase type
        matth_coef = matthews_corrcoef(true_labels_np[:,label_idx], predicted_labels_np[:,label_idx])
        matthews_coefficients.append(matth_coef)

    # Calculate the average accuracy over all labels
    accuracy = np.mean(accuracies)
    matthews_coefficient = np.mean(matthews_coefficients)
    model.train()
    return accuracy, matthews_coefficient


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--checkpoint_file", type=str, default="bart_model.ckpt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--multi_label", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    return args


def finetune_paraphrase_detection(args):
    model = BartWithClassifier()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep="\t")
    quora_train_data = []
    with open("data/quora-paraphrase-train.csv", "r", encoding="utf-8") as fp:
        for record in csv.DictReader(fp, delimiter="\t"):
            try:
                sent_id = record["id"].lower().strip()
                quora_train_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        int(float(record["is_duplicate"])),
                        sent_id,
                    )
                )
            except:
                pass
    quora_train_data = select_subset(quora_train_data, 5)
    quora_train_data = SentencePairDataset(quora_train_data, args)


    # Split train data into train and validation sets
    train_val_split = int(0.9 * len(train_dataset))
    train_data = train_dataset.iloc[:train_val_split]
    val_data = train_dataset.iloc[train_val_split:]

    train_dataloader = transform_data(train_data, args)
    val_dataloader = transform_data(val_data, args)
    test_dataloader = transform_data(test_dataset, args)

    quora_train_dataloader = DataLoader(
        quora_train_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=quora_train_data.collate_fn,
    )

    print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples.")

    early_stopping = EarlyStopping(args.checkpoint_file, patience=4)
    model = train_model(model, train_dataloader, quora_train_dataloader, val_dataloader, device, args, early_stopping)

    # torch.save(model.state_dict(), args.checkpoint_file)
    print(f"Training finished. Saved model at {args.checkpoint_file}")

    model.load_state_dict(torch.load(args.checkpoint_file))
    accuracy, matthews_corr = evaluate_model(model, val_dataloader, device)
    print(f"The accuracy of the model is: {accuracy:.3f}")
    print(f"Matthews Correlation Coefficient of the model is: {matthews_corr:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_dataloader, test_ids, device)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False, sep="\t"
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)