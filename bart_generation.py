import random, numpy as np, argparse, pandas as pd
import torch
from transformers import BartForConditionalGeneration
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AdamW
from sacrebleu.metrics import BLEU
    

def transform_data(dataset, max_length=256):
    '''
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + Tokenized sentence_1 segment location + SEP + Tokenized paraphrase types.
    Return Data Loader.
    '''
    ### TODO
    raise NotImplementedError


def train_model(model, train_data, device):
    '''
    Train the model. Return and save the model.
    '''
    ### TODO
    raise NotImplementedError



def test_model(test_data, device, model):
    '''
    Test the model. Predict the results and return them as the Pandas dataframe with columns 'id' and 'sentence2' (those are the generations of the model for given sentence1). 
    Format of the data in the columns should be the same as in the train dataset. 
    Return this dataframe.
    '''
    ### TODO
    raise NotImplementedError

def evaluate_model(model, test_data, device, tokenizer):
    '''
    You can use your train/validation set to evaluate models performance with the BLEU score.
    '''
    bleu = BLEU()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_beams=5, early_stopping=True)
            
            pred_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in outputs]
            ref_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in labels]
            
            predictions.extend(pred_text)
            references.extend([[r] for r in ref_text]) 
            
    bleu_score = bleu.corpus_score(predictions, references)
    return bleu_score.score

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


def finetune_paraphrase_generation(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    model_name = 'facebook/bart-large'
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    
    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv")
    
    # You might do a split of the train data into train/validation set here
    #...
    
    train_data = transform_data(train_dataset)
    test_data = transform_data(test_dataset)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    bleu_score = evaluate_model(model,train_data,device,tokenizer)
    print("The BLEU-score of the model is: ", bleu_score)
    
    test_results = test_model(test_data, device, model)
    test_results.to_csv("predictions/etpc-paraphrase-generation-test-out.csv" , index=False)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
