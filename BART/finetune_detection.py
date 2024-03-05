import random, numpy as np, argparse, pandas as pd
import torch
from transformers import BartModel
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AdamW
from sklearn.metrics import accuracy_score


class BartWithClassifier(nn.Module):
    def __init__(self, model_name="facebook/bart-large", num_labels=7):
        super(BartWithClassifier, self).__init__()
        self.bart = BartModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output) 
        probabilities = self.sigmoid(logits)
        return probabilities
    

def transform_data(dataset, max_length=512):
    '''
    Turn the data to the format you want to use. 
    Use sentences segment location and sentences.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Turn labels to the binary form (ex. [2, 5, 6, 0, 0, 0, 0] -> [0, 1, 0, 0, 1, 1, 0]).
    Return Data Loader.
    '''
    raise NotImplementedError


def train_model(model, train_data, device):
    '''
    Train the model. Return the model.
    '''
    ### TODO
    raise NotImplementedError


def test_model(model, test_data, device):
    '''
    Test the model. Predict the results and return them as the Pandas dataframe with columns 'id' and 'paraphrase_types'.
    The 'paraphrase_type' column should contain the binary array of your model predictions. 
    Return this dataframe.
    '''
    ### TODO
    
    raise NotImplementedError

def evaluate_model(model, test_data, device):
    '''
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the seven paraphrase types the model has predicted correctly for each data point. 
    So, if the models prediction is [1,1,0,0,1,1,0] and the true label is [0,0,0,0,1,1,0], this predicition
    has an accuracy of 5/7, i.e. 71.4% .
    '''
    all_pred = []
    all_labels = []
    model.eval()
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
    
    accuracies = []
    for label_idx in range(true_labels_np.shape[1]):
        label_accuracy = accuracy_score(true_labels_np[:, label_idx], predicted_labels_np[:, label_idx])
        accuracies.append(label_accuracy)
    
    accuracy = np.mean(accuracies)
    return accuracy

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
    parser.add_argument("--use_gpu", action='store_true')
    args = parser.parse_args()
    return args


def finetune_paraphrase_detection(args):
    model = BartWithClassifier()
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    model.to(device)
    
    train_dataset = pd.read_csv('Datasets/paraphrase-train.csv')
    test_dataset = pd.read_csv('Datasets/paraphrase-detection-test-student.csv')
    # You might do a split of the train data into train/validation set here
    #...
    
    train_data = transform_data(train_dataset)
    test_data = transform_data(test_dataset)
    
    model = train_model(model, train_data, device)
    
    accuracy = evaluate_model(model,train_data, device)
    print("The accuracy of the Model is: ", accuracy)
    
    test_results = test_model(model, test_data, device)
    test_results.to_csv('Predictions/detection-test-out.csv', index=False)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
