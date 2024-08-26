import argparse
import os
from pprint import pformat
import random
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW
from losses import FocalLoss, DiceLoss
TQDM_DISABLE = False

# Function to select a subset
def select_subset(dataset, percentage):
    subset_size = int(len(dataset) * (percentage / 100))
    subset_indices = random.sample(range(len(dataset)), subset_size)
    subset = [dataset[i] for i in subset_indices]
    return subset


# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
DROPOUT = 0.0
WEIGHTDECAY = 0.0
BATCHNORM = False
L1_LAMBDAl = 0
GradientClipping = False
LABELSMOOTHING = 0.0


# loss function
# CrossEntropyLoss (cel), FocalLoss (fl), Hinge Loss (Multi-Class SVM) (hl),
# Mean Squared Error (MSE) for Soft Labels (mse), Dice Loss (Soft Dice Loss) (dl).
LOSS = "fl"

class MultitaskBERT(nn.Module):
    """
    This module should use BERT for these tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    (- Paraphrase type detection (predict_paraphrase_types))
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()

        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert parameters.
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True
        ### TODO
        # a linear layer for paraphrase detection
        self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.sts_head = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        
        # raise NotImplementedError
        # raise NotImplementedError
        self.dropout = nn.Dropout(p=DROPOUT)
        self.batch_norm = nn.BatchNorm1d(BERT_HIDDEN_SIZE) 
        self.sentiment_linear = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.biary_sentiment_linear = nn.Linear(BERT_HIDDEN_SIZE, 2)


    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""

        # The final BERT embedding is the hidden state of [CLS] token (the first token).
        # See BertModel.forward() for more details.
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        bert_model = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output_layer = bert_model['last_hidden_state'][:, 0, :]
        return output_layer
        # ### TODO
        # raise NotImplementedError

    def predict_sentiment(self, input_ids, attention_mask):
        """
        Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        Dataset: SST
        """
        ### TODO
        # raise NotImplementedError
        bert_output = self.forward(input_ids, attention_mask)
        bert_output = self.dropout(bert_output)
        if BATCHNORM: 
            bert_output = self.batch_norm(bert_output)
        sentiment_out = self.sentiment_linear(bert_output)
        return sentiment_out

    def predict_binary_sentiment(self, input_ids, attention_mask):
        """
        Given a batch of sentences, outputs logits for classifying sentiment.
        There are 2 sentiment classes:
        (0 - negative, 1- positive)
        Thus, your output should contain 2 logits for each sentence.
        Dataset: IMDB
        """
        ### TODO
        # raise NotImplementedError
        bert_output = self.forward(input_ids, attention_mask)
        sentiment_out = self.biary_sentiment_linear(bert_output)
        return sentiment_out



    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: Quora
        """
        ### TODO
        # Get embeddings for both sentences
        embedding_1 = self.forward(input_ids_1, attention_mask_1)
        embedding_2 = self.forward(input_ids_2, attention_mask_2)

        # Concatenate the embeddings
        combined_embedding = torch.cat((embedding_1, embedding_2), dim=1)

        # Pass through the paraphrase classifier
        logit = self.paraphrase_classifier(combined_embedding)

        return logit.squeeze(-1)

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Since the similarity label is a number in the interval [0,5], your output should be normalized to the interval [0,5];
        it will be handled as a logit by the appropriate loss function.
        Dataset: STS
        """
        # Get both Embeddings
        output_1 = self.forward(input_ids_1, attention_mask_1)
        output_2 = self.forward(input_ids_2, attention_mask_2)
        
        # Compute cosine similarity between both sentence embeddings
        cos_sim = F.cosine_similarity(output_1, output_2)

        # Scale similarity to match labels between [0,5]
        logit = (cos_sim + 1) * 2.5
        return logit

    def eval_fn(self,perturbed_embeddings_1,perturbed_embeddings_2):
        # Compute cosine similarity between both sentence embeddings
        cos_sim = F.cosine_similarity(perturbed_embeddings_1, perturbed_embeddings_2)

        # Scale similarity to match labels between [0,5]
        logit = (cos_sim + 1) * 2.5
        return logit
    
    def get_embeddings(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        output_1 = self.forward(input_ids_1, attention_mask_1)
        output_2 = self.forward(input_ids_2, attention_mask_2)
        
        return output_1,output_2#    

    def compute_multiple_negatives_ranking_loss(self, embeddings_1, embeddings_2):
        # Normalize the embeddings to unit vectors
        embeddings_1 = F.normalize(embeddings_1, p=2, dim=1)
        embeddings_2 = F.normalize(embeddings_2, p=2, dim=1)

        # Compute cosine similarity matrix between the two sets of embeddings
        sim_mat = torch.matmul(embeddings_1, embeddings_2.T)

        # Apply softmax to similarity matrix (to ensure it's positive)
        sim_mat = sim_mat * 20.0  # Optional scaling factor (as in the original repo)
        labels = torch.arange(sim_mat.size(0)).to(embeddings_1.device)

        # Compute the cross-entropy loss
        loss = F.cross_entropy(sim_mat, labels)

        return loss
    

    def predict_paraphrase_types(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Given a batch of pairs of sentences, outputs logits for detecting the paraphrase types.
        There are 7 different types of paraphrases.
        Thus, your output should contain 7 unnormalized logits for each sentence. It will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: ETPC
        """
        ### TODO
        raise NotImplementedError

def inf_norm(x):
    return torch.norm(x, p=2, dim=-1, keepdim=True)

def smart_loss(model, embeddings_1,embeddings_2, logits, epsilon=1e-3):
    # Generate perturbed embeddings
    embed_norm_1= inf_norm(embeddings_1 )
    noise_1 = torch.randn_like(embeddings_1) * epsilon * embed_norm_1
    perturbed_embeddings_1 = embeddings_1 + noise_1

    embed_norm_2 = inf_norm(embeddings_2)
    noise_2 = torch.randn_like(embeddings_2) * epsilon * embed_norm_2
    perturbed_embeddings_2 = embeddings_2 + noise_2

    # Get predictions for perturbed embeddings
    perturbed_logits = model.eval_fn(perturbed_embeddings_1,perturbed_embeddings_2)

    # Compute KL divergence
    kl_div = nn.KLDivLoss(reduction="batchmean")

    return kl_div(F.log_softmax(perturbed_logits, dim=-1), F.softmax(logits, dim=-1))


def save_model(model, optimizer, args, config, filepath):
    # ---------------------------------------------------
    # first run wherethe directory models doesn't exist
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # ---------------------------------------------------
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"Saving the model to {filepath}.")


# TODO Currently only trains on SST dataset!
def train_multitask(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load data
    # Create the data and its corresponding datasets and dataloader:
    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data, tweets_train_data= load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, args.tweets_train, split="train"
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data, tweets_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, args.tweets_dev, split="train"
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    tweets_train_dataloader = None
    tweets_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None

    # SST dataset
    if args.task == "sst" or args.task == "multitask" or args.task == "multi-sentiment" or args.task == "sas":
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_train_dataloader = DataLoader(
            sst_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sst_train_data.collate_fn,
        )
        sst_dev_dataloader = DataLoader(
            sst_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sst_dev_data.collate_fn,
        )

    # tweets dataset
    if args.task == "tweets" or args.task == "multitask" or args.task == "multi-sentiment":
        tweets_train_data = SentenceClassificationDataset(tweets_train_data, args)
        tweets_dev_data = SentenceClassificationDataset(tweets_dev_data, args)

        tweets_train_dataloader = DataLoader(
            tweets_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=tweets_train_data.collate_fn,
        )
        tweets_dev_dataloader = DataLoader(
            tweets_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=tweets_dev_data.collate_fn,
        )   

    # Quora dataset (Paraphrase Detection)
    if args.task == "qqp" or args.task == "multitask":
        quora_train_data = SentencePairDataset(quora_train_data, args)
        quora_dev_data = SentencePairDataset(quora_dev_data, args)
        
        # subset_percentage = 5
        # # Select subsets
        # quora_train_data_subset = select_subset(quora_train_data, subset_percentage)
        # quora_dev_data_subset = select_subset(quora_dev_data, subset_percentage)

        # # Create new SentencePairDataset instances with the subsets
        # quora_train_data = SentencePairDataset(quora_train_data_subset, args)
        # quora_dev_data = SentencePairDataset(quora_dev_data_subset, args)

        quora_train_dataloader = DataLoader(
            quora_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=quora_train_data.collate_fn,
        )
        quora_dev_dataloader = DataLoader(
            quora_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=quora_dev_data.collate_fn,
        )
    
    # STS dataset
    if args.task == "sts" or args.task == "multitask" or args.task == "sas":
        sts_train_data = SentencePairDataset(sts_train_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args)

        sts_train_dataloader = DataLoader(
            sts_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sts_train_data.collate_fn,
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sts_dev_data.collate_fn,
        )

    ### TODO
    #   Load data for the other datasets
    # If you are doing the paraphrase type detection with the minBERT model as well, make sure
    # to transform the the data labels into binaries (as required in the bart_detection.py script)

    # Init model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
    }

    config = SimpleNamespace(**config)

    separator = "-" * 30
    print(separator)
    print("    BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    
    
    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHTDECAY)
    best_dev_acc = float("-inf")

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        if args.task == "sst" or args.task == "multitask" or args.task == "multi-sentiment" or args.task == "sas":
            # Train the model on the sst dataset.

            for batch in tqdm(
                sst_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids, b_mask, b_labels = (
                    batch["token_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                
                # choose the loss function
                if LOSS == "cel":
                    loss = F.cross_entropy(logits, b_labels.view(-1), label_smoothing=LABELSMOOTHING)
                if LOSS == 'fl':
                    loss = FocalLoss(gamma=2.0)(logits, b_labels.view(-1))
                if LOSS == 'hl':
                    loss = nn.MultiMarginLoss()(logits, b_labels.view(-1))
                if LOSS == "mse":
                    smoothed_labels = (1 - LABELSMOOTHING) * F.one_hot(b_labels, num_classes=N_SENTIMENT_CLASSES) + LABELSMOOTHING / N_SENTIMENT_CLASSES
                    loss = F.mse_loss(F.softmax(logits, dim=-1), smoothed_labels)
                if LOSS == "dl":
                    loss_fn = DiceLoss()
                    targets = F.one_hot(b_labels, num_classes=N_SENTIMENT_CLASSES)
                    loss = loss_fn(logits, targets)
                
                # L1 loss
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + L1_LAMBDAl * l1_norm
                
                # Gradient Clipping
                if GradientClipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
        
        if args.task == "tweets" or args.task == "multitask" or args.task == "multi-sentiment":
            # Train the model on the tweets dataset.

            for batch in tqdm(
                tweets_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids, b_mask, b_labels = (
                    batch["token_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "sts" or args.task == "multitask" or args.task == "sas":
            # Trains the model on the sts dataset
            for batch in tqdm(
                sts_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                input_ids_1, attention_mask_1, input_ids_2, attention_mask_2 ,labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                    )

                input_ids_1 = input_ids_1.to(device)
                attention_mask_1 = attention_mask_1.to(device)
                input_ids_2 = input_ids_2.to(device)
                attention_mask_2 = attention_mask_2.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                logits = model.predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)                
                mse_loss = F.mse_loss(logits, labels.float())

                # Get embeddings from the model for SMART loss computation
                embeddings_1,embeddings_2 = model.get_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                s_loss = smart_loss(model , embeddings_1, embeddings_2, logits)

                # mnrl loss
                ranking_loss = model.compute_multiple_negatives_ranking_loss(embeddings_1,embeddings_2)
                loss = ranking_loss + 0.02 * s_loss + mse_loss

                # Backpropagation and optimization step
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            # ### TODO
            # raise NotImplementedError

        if args.task == "qqp" or args.task == "multitask":
            # Trains the model on the qqp dataset
            ### TODO
            # Train the model on the Quora dataset (Paraphrase Detection)
            for batch in tqdm(
                quora_train_dataloader, desc=f"train-qqp-{epoch+1:02}", disable=TQDM_DISABLE
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

                train_loss += loss.item()
                num_batches += 1
            
        if args.task == "etpc" or args.task == "multitask":
            # Trains the model on the etpc dataset
            ### TODO
            raise NotImplementedError

        train_loss = train_loss / num_batches

        quora_train_acc, _, _, sst_train_acc, _, _, tweets_train_acc, _, _, sts_train_corr, _, _, etpc_train_acc, _, _ = (
            model_eval_multitask(
                sst_train_dataloader,
                quora_train_dataloader,
                sts_train_dataloader,
                etpc_train_dataloader,
                tweets_train_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        quora_dev_acc, _, _, sst_dev_acc, _, _, tweets_dev_acc, _, _, sts_dev_corr, _, _, etpc_dev_acc, _, _ = (
            model_eval_multitask(
                sst_dev_dataloader,
                quora_dev_dataloader,
                sts_dev_dataloader,
                etpc_dev_dataloader,
                tweets_dev_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )
        print('SST', sst_train_acc,sst_dev_acc)
        print('tweets', tweets_train_acc, tweets_dev_acc)
        train_acc, dev_acc = {
            "sst": (sst_train_acc, sst_dev_acc),
            "tweets": (tweets_train_acc, tweets_dev_acc),
            "sts": (sts_train_corr, sts_dev_corr),
            "qqp": (quora_train_acc, quora_dev_acc),
            "etpc": (etpc_train_acc, etpc_dev_acc),
            "multi-sentiment": (sst_train_acc, sst_dev_acc),  
            "sas": (sst_train_acc, sst_dev_acc),  
            "multitask": (0, 0),  # TODO
        }[args.task]

        print(
            f"Epoch {epoch+1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)


def test_model(args, filepath = None):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        if filepath != None:
            saved = torch.load(filepath)
        else:
            saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"], strict=False)
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        return test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()

    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='choose between "sst","sts","qqp","etpc", "tweets", "sas", "multitask" to train for different tasks ',
        choices=("sst", "sts", "qqp", "etpc", "multitask", "tweets", "multi-sentiment", "sas"),
        default="sst",
    )

    # Model configuration
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )
    parser.add_argument("--use_gpu", action="store_true")

    args, _ = parser.parse_known_args()

    # Dataset paths
    parser.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/sst-sentiment-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/sst-sentiment-test-student.csv")
    
    parser.add_argument("--tweets_train", type=str, default="data/processed_tweets_train.csv")
    parser.add_argument("--tweets_dev", type=str, default="data/processed_tweets_dev.csv")
    parser.add_argument("--tweets_test", type=str, default="data/processed_tweets_test.csv")

    parser.add_argument("--quora_train", type=str, default="data/quora-paraphrase-train.csv")
    parser.add_argument("--quora_dev", type=str, default="data/quora-paraphrase-dev.csv")
    parser.add_argument("--quora_test", type=str, default="data/quora-paraphrase-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-similarity-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-similarity-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-similarity-test-student.csv")

    # TODO
    # You should split the train data into a train and dev set first and change the
    # default path of the --etpc_dev argument to your dev set.
    parser.add_argument("--etpc_train", type=str, default="data/etpc-paraphrase-train-split.csv") # CHANGE ME BACK - REMOVE
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-dev-split.csv") # CHANGE ME BACK - REMOVE
    parser.add_argument(
        "--etpc_test", type=str, default="data/etpc-paraphrase-detection-test-student.csv"
    )

    # Output paths
    parser.add_argument(
        "--sst_dev_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sst_test_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-test-output.csv"
        ),
    )

    parser.add_argument(
        "--quora_dev_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--quora_test_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-test-output.csv"
        ),
    )

    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-test-output.csv"
        ),
    )

    parser.add_argument(
        "--etpc_dev_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--etpc_test_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-test-output.csv"
        ),
    )

    # Hyperparameters
    parser.add_argument("--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    parser.add_argument("--local_files_only", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"models/{args.option}-{args.epochs}-{args.lr}-{LOSS}-{'GradientClipping-' if GradientClipping else ''}{'BatchNorm-' if BATCHNORM else ''}dr({DROPOUT})-l1({L1_LAMBDAl})-wd({WEIGHTDECAY})-LS({LABELSMOOTHING})-{args.task}.pt"  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    # filepath = "/user/ahmed.assy/u11454/old_project/models/sst/finetune-10-1e-05-dr-0.0-wd-0.0-focal-sst.pt-->53%"
    test_model(args)
