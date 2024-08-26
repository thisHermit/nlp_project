import argparse
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from optimizer import AdamW

import os
import sys

import torch.nn.functional as F

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory
os.chdir(script_dir)

# Add the script's directory to the Python path
sys.path.insert(0, script_dir)

# Print the current working directory to verify
print("Current working directory:", os.getcwd())

TQDM_DISABLE = False


def transform_data(dataset, args, max_length=256, shuffle=True):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase types.
    Return Data Loader.
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for _, row in dataset.iterrows():
        sentence1 = row['sentence1']
        sentence1_segment = ' '.join(map(str, eval(row['sentence1_segment_location'])))
        paraphrase_types = ' '.join(map(str, eval(row['paraphrase_types'])))
        
        input_text = f"{sentence1} </s> {sentence1_segment} </s> {paraphrase_types}"
        target_text = row['sentence2'] if 'sentence2' in row else ""
        
        inputs = tokenizer(input_text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        targets = tokenizer(target_text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        
        input_ids.append(inputs['input_ids'].squeeze())
        attention_masks.append(inputs['attention_mask'].squeeze())
        labels.append(targets['input_ids'].squeeze())
    
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(labels)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    
    return dataloader


def cosine_embedding_loss(model, tokenizer, generated_ids, target_ids, device):
    # Get the encoder's output for the generated sentence
    generated_attention_mask = (generated_ids != tokenizer.pad_token_id).long()
    generated_encoder_output = model.get_encoder()(generated_ids, attention_mask=generated_attention_mask).last_hidden_state

    # Get the encoder's output for the target sentence
    target_attention_mask = (target_ids != tokenizer.pad_token_id).long()
    target_encoder_output = model.get_encoder()(target_ids, attention_mask=target_attention_mask).last_hidden_state

    # Calculate cosine similarity between the mean pooled representations
    generated_embedding = torch.mean(generated_encoder_output, dim=1)
    target_embedding = torch.mean(target_encoder_output, dim=1)
    
    return F.cosine_embedding_loss(generated_embedding, target_embedding, torch.ones(generated_embedding.size(0)).to(device))

def identity_loss(input_ids, generated_ids, tokenizer):
    """
    Calculate the identity loss between input and generated sentences.
    Returns 1 if the smaller text appears within the larger text, 0 otherwise.
    """
    input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    losses = []
    for i_text, g_text in zip(input_text, generated_text):
        i_text = i_text.strip().lower()
        g_text = g_text.strip().lower()
        
        if len(i_text) <= len(g_text):
            smaller, larger = i_text, g_text
        else:
            smaller, larger = g_text, i_text
        
        if smaller in larger:
            losses.append(1.0)
        else:
            losses.append(0.0)
    
    return torch.tensor(losses, device=input_ids.device).mean()

def train_model(model, train_data, train_dataset, dev_data, args, device, tokenizer):
    """
    Train the model. Return and save the model.
    """
    ### TODO
    try:
        optimizer = AdamW(model.parameters(), lr=5e-5)
        num_epochs = args.epochs
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_data, disable=TQDM_DISABLE):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                 # Generate output from the model
                generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256)
                
                # Decode the input and generated sequences to inspect them
                input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
    
                # Calculate identity loss
                id_loss = identity_loss(input_ids, generated_ids, tokenizer)
                cos_loss = cosine_embedding_loss(model, tokenizer, generated_ids, labels, device)
                
                # Combine losses
                loss = outputs.loss + cos_loss + id_loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_data)
            train_score = evaluate_model(model, train_dataset, args, device, tokenizer)
            # Evaluate on dev set
            dev_score = evaluate_model(model, dev_data, args, device, tokenizer)
            print(f"Epoch {epoch+1}/{num_epochs}, Average training loss: {avg_train_loss:.4f}")
            print(f"Train score: {train_score:.4f}")
            print(f"Dev score: {dev_score:.4f}")
            print()
        
        return model
    except Exception as e:
        print(f"Error in train_model: {e}")
        raise e


def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Test the model. Generate paraphrases for the given sentences (sentence1) and return the results
    in form of a Pandas dataframe with the columns 'id' and 'Generated_sentence2'.
    The data format in the columns should be the same as in the train dataset.
    Return this dataframe.
    """
    ### TODO
    try:
        model.eval()
        generated_paraphrases = []
        
        with torch.no_grad():
            for batch in tqdm(test_data, disable=TQDM_DISABLE):
                try:
                    input_ids, attention_mask, _ = [b.to(device) for b in batch]
                    
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=256,
                        num_beams=5,
                        early_stopping=True
                    )
                    
                    decoded_outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
                    generated_paraphrases.extend(decoded_outputs)
                except Exception as e:
                    print(f"Error processing test batch: {e}")
                    continue
        
        results_df = pd.DataFrame({
            'id': test_ids,
            'Generated_sentence2': generated_paraphrases
        })
        
        return results_df
    except Exception as e:
        print(f"Error in test_model: {e}")
        raise e


def evaluate_model(model, test_data, args, device, tokenizer, verbose=False):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    test_data is a Pandas Dataframe, the column "sentence1" contains all input sentence and 
    the column "sentence2" contains all target sentences
    """
    model.eval()
    bleu = BLEU()
    predictions = []

    dataloader = transform_data(test_data, args, shuffle=False)
    with torch.no_grad():
        for batch in dataloader: 
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )
            
            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            
            predictions.extend(pred_text)

    inputs = test_data["sentence1"].tolist()
    references = test_data["sentence2"].tolist()

    model.train()
    # Calculate BLEU score
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    # Penalize BLEU score if its to close to the input
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score

    if verbose:
        print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")
    

    # Penalize BLEU and rescale it to 0-100
    # If you perfectly predict all the targets, you should get an penalized BLEU score of around 52
    penalized_bleu = bleu_score_reference*bleu_score_inputs/ 52
    if verbose:
        print(f"Penalized BLEU Score: {penalized_bleu}")

    return penalized_bleu


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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train-split.csv", sep="\t")
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev-split.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv", sep="\t")

    # You might do a split of the train data into train/validation set here
    # ...

    train_data = transform_data(train_dataset, args)
    dev_data = transform_data(dev_dataset, args)
    test_data = transform_data(test_dataset, args, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples.")

    print("Loading model pre-trained on PAWS...")
    model.load_state_dict(torch.load('paws_bart_generation_model.ckpt'))
    model = train_model(model, train_data, train_dataset, dev_dataset, args, device, tokenizer)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_dataset, args, device, tokenizer)
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False, sep="\t"
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)