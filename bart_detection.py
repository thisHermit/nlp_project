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
import optuna

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
    def __init__(self, args, num_labels=7):
        super(BartWithClassifier, self).__init__()

        latent_dims = args.latent_dims

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=True, add_cross_attention=True)
        # self.pre_final = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)
        self.fc_mu = nn.Linear(self.bart.config.hidden_size, latent_dims) # Linear layer for mu
        self.fc_logvar = nn.Linear(self.bart.config.hidden_size, latent_dims) # Linear layer for log variance
        self.fc_z = nn.Linear(latent_dims, self.bart.config.hidden_size)

        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, input_ids, attention_mask=None):
        # use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # add two fully connected layers to obtain the logits
        mu = self.fc_mu(cls_output)
        logvar = self.fc_logvar(cls_output)

        z = self.reparameterize(mu, logvar) if self.training else mu

        z_out = self.fc_z(z)
        
        out = self.classifier(z_out + cls_output) # add residual

        return out


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
        binary_labels = [[1 if (i + 1) in label else 0 for i in range(7)] for label in labels]
        labels_tensor = torch.tensor(binary_labels, dtype=torch.float32)
        
        data = TensorDataset(input_ids, attention_mask, labels_tensor)
    else:
        data = TensorDataset(input_ids, attention_mask)
    
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True if 'paraphrase_types' in dataset.columns else False)
    
    return dataloader


def train_model(model, train_data, dev_data, device, args, early_stopping=None):
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(args.epochs):
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
        
        scheduler.step()

        avg_loss = total_loss / len(train_data)
        train_accuracy = correct_predictions / total_predictions
        dev_accuracy, dev_matthews_coefficient = evaluate_model(model, dev_data, device)
        total_train_accuracy, total_train_matthews_coefficient = evaluate_model(model, train_data, device)

        print(f"Epoch {epoch+1}/{args.epochs}")
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
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--scheduler_gamma", type=float, default=0.9)
    parser.add_argument("--train_val_split_ratio", type=float, default=0.9)
    parser.add_argument("--optuna_optim", action="store_true")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--latent_dims", type=int, default=7)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    return args


def finetune_paraphrase_detection(args):
    model = BartWithClassifier()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep="\t")

    # Split train data into train and validation sets
    train_val_split = int(0.9 * len(train_dataset))
    train_data = train_dataset.iloc[:train_val_split]
    val_data = train_dataset.iloc[train_val_split:]

    train_dataloader = transform_data(train_data, args)
    val_dataloader = transform_data(val_data, args)
    test_dataloader = transform_data(test_dataset, args)

    print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples.")

    early_stopping = EarlyStopping(args.checkpoint_file, patience=4)
    model = train_model(model, train_dataloader, val_dataloader, device, args, early_stopping)

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

def optuna_objective(trial):
    args.epochs = trial.suggest_int("epochs", 3, 40)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e-5)
    args.scheduler_gamma = trial.suggest_uniform("scheduler_gamma", 0.8, 0.99)
    args.latent_dims = trial.suggest_int("latent_dims", 4, 14)
    args.patience = trial.suggest_int("patience", 2, 8)

    model = BartWithClassifier(args)
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")

    train_val_split = int(args.train_val_split_ratio * len(train_dataset))
    train_data = train_dataset.iloc[:train_val_split]
    val_data = train_dataset.iloc[train_val_split:]

    train_dataloader = transform_data(train_data, args)
    val_dataloader = transform_data(val_data, args)

    early_stopping = EarlyStopping(args.checkpoint_file, patience=args.patience)
    
    model = train_model(model, train_dataloader, val_dataloader, device, args, early_stopping)

    model.load_state_dict(torch.load(args.checkpoint_file))
    _, matthews_corr = evaluate_model(model, val_dataloader, device)
    return matthews_corr

def optuna_search(args):
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Use the best hyperparameters for final training
    for key, value in trial.params.items():
        setattr(args, key, value)
    
    finetune_paraphrase_detection(args)

def finetune_paraphrase_detection(args):
    model = BartWithClassifier(args)
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep="\t")

    train_val_split = int(args.train_val_split_ratio * len(train_dataset))
    train_data = train_dataset.iloc[:train_val_split]
    val_data = train_dataset.iloc[train_val_split:]

    train_dataloader = transform_data(train_data, args)
    val_dataloader = transform_data(val_data, args)
    test_dataloader = transform_data(test_dataset, args)

    print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples.")

    early_stopping = EarlyStopping(args.checkpoint_file, patience=args.patience)
    model = train_model(model, train_dataloader, val_dataloader, device, args, early_stopping)

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
    if args.optuna_optim:
        optuna_search(args)
    else:
        finetune_paraphrase_detection(args)
