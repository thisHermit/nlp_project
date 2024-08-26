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

import warnings

from bert import BertModel
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import get_optimizer

from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler

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

class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, embeddings, attention_mask):
        # Calculate the attention scores
        attention_scores = self.attention_weights(embeddings).squeeze(-1)
        # Apply the mask to ignore padding tokens
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        # Calculate the attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)
        # Multiply the attention weights with the embeddings
        weighted_embeddings = embeddings * attention_weights.unsqueeze(-1)
        # Sum the weighted embeddings to get the sentence embedding
        sentence_embedding = weighted_embeddings.sum(dim=1)
        return sentence_embedding
    

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob):
        super(MultiHeadAttentionLayer, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, embeddings, attention_mask=None):
        # embeddings: (batch_size, seq_length, hidden_size)
        # attention_mask: (batch_size, seq_length)
        
        # Prepare for multi-head attention (requires batch_first=False)
        embeddings = embeddings.transpose(0, 1)  # (seq_length, batch_size, hidden_size)
        
        # Apply multi-head attention
        attention_output, _ = self.mha(embeddings, embeddings, embeddings, key_padding_mask=attention_mask)
        
        # Apply residual connection
        attention_output = attention_output + embeddings
        
        # Apply dropout and layer normalization
        attention_output = self.layer_norm(self.dropout(attention_output))
        
        # Return the output in the original shape (batch_size, seq_length, hidden_size)
        return attention_output.transpose(0, 1)

    
# Default MHA Configuration
mha_config = {
    'num_heads': 8,
    'dropout_prob': 0.3,
}

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob, num_layers=3):
        super(MLPHead, self).__init__()
        self.layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < num_layers - 1:  # No activation or normalization after last layer
                self.layers.append(nn.LayerNorm(dims[i+1]))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout_prob))
                
                # Add residual connection if dimensions match
                if dims[i] == dims[i+1]:
                    self.residual_layers.append(nn.Identity())
                else:
                    self.residual_layers.append(nn.Linear(dims[i], dims[i+1]))

    def forward(self, x):
        residual = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % 4 == 3 and i < len(self.layers) - 1:  # After each block except the last
                x = x + self.residual_layers[i // 4](residual)
                residual = x
        return x  

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
        
        self.num_layers = len(self.bert.bert_layers)
        
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True
        
        if args.scheduler == 'gradual_unfreeze':
            # Initialize layer freezing status
            self.frozen_layers = [True] * self.num_layers
        
            # Freeze all layers initially
            self.freeze_all_layers()
            
                
        self.pooling_type = args.pooling_type  # 'cls', 'mean', or 'max'
        if self.pooling_type == 'self_attention':
            self.self_attention_pooling = SelfAttentionPooling(BERT_HIDDEN_SIZE)
            
        self.use_mha = args.use_mha
        # Multi-Head Attention Layer
        if self.use_mha:
            self.mha_layer = MultiHeadAttentionLayer(hidden_size=BERT_HIDDEN_SIZE, num_heads=mha_config['num_heads'], dropout_prob=mha_config['dropout_prob'])
        
        ### TODO
        # MLP Head with Dropout and Residual Connections
        # MLP Head with Dropout and Residual Connections
        if args.use_mlp:
            self.paraphrase_classifier = MLPHead(input_dim=BERT_HIDDEN_SIZE * 2, hidden_dim=BERT_HIDDEN_SIZE, output_dim=1, dropout_prob=0.3)
        else:
            self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        
        self.sts_head = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE * 2, BERT_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(BERT_HIDDEN_SIZE, 1)
        )
        # raise NotImplementedError
        # raise NotImplementedError
        self.sentiment_linear = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        
        # Initialize weights for the combined loss for the model
        self.weight_cosine, self.weight_bce, self.weight_ranking = (0.33, 0.33, 0.34)
        
        # SMART Loss setup
        self.smart_loss = SMARTLoss(
            eval_fn=self.smart_eval_fn,
            loss_fn=F.binary_cross_entropy_with_logits,
            loss_last_fn=sym_kl_loss,
            step_size=1e-3,
            num_steps=1
        )
        
    
    def smart_eval_fn(self, embed):
        """ 
        Evaluation function for SMART loss.
        Here, we assume the evaluation function returns the output logits for paraphrase detection.
        """
        logits = self.paraphrase_classifier(embed)
        return logits.squeeze(-1)


    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""

        # The final BERT embedding is the hidden state of [CLS] token (the first token).
        # See BertModel.forward() for more details.
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        bert_model = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if args.task == "qqp":
            return bert_model
            
        output_layer = bert_model['last_hidden_state'][:, 0, :]
        return output_layer
        # ### TODO
        # raise NotImplementedError
        """Takes a batch of sentences and produces embeddings for them."""

        # The final BERT embedding is the hidden state of [CLS] token (the first token).
        # See BertModel.forward() for more details.
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        # ### TODO
        # raise NotImplementedError
        
    def get_sentence_embeddings(self, input_ids, attention_mask):
        bert_output = self.forward(input_ids, attention_mask)
        
        if self.pooling_type == 'cls':
            return bert_output['last_hidden_state'][:, 0, :]
        elif self.pooling_type == 'mean':
            return (bert_output['last_hidden_state']* attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooling_type == 'max':
            return torch.max(bert_output['last_hidden_state'] * attention_mask.unsqueeze(-1), dim=1)[0]
        elif self.pooling_type == 'self_attention':
            return self.self_attention_pooling(bert_output['last_hidden_state'], attention_mask)
        else:
            raise ValueError("Invalid pooling type. Choose 'cls', 'mean', 'max', or 'self_attention'.")

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
        sentiment_out = self.sentiment_linear(bert_output)
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
        embedding_1 = self.get_sentence_embeddings(input_ids_1, attention_mask_1)
        embedding_2 = self.get_sentence_embeddings(input_ids_2, attention_mask_2)
        
        if self.use_mha:
            embedding_1 = self.mha_layer(embedding_1.unsqueeze(1)).squeeze(1)
            embedding_2 = self.mha_layer(embedding_2.unsqueeze(1)).squeeze(1)
        
        # Concatenate embeddings and pass through linear head
        combined_embedding = torch.cat((embedding_1, embedding_2), dim=1)
        logit = self.paraphrase_classifier(combined_embedding)

        return logit.squeeze(-1)

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Since the similarity label is a number in the interval [0,5], your output should be normalized to the interval [0,5];
        it will be handled as a logit by the appropriate loss function.
        Dataset: STS
        """
        # print(input_ids_1)
        output_1 = self.forward(input_ids_1, attention_mask_1)
        output_2 = self.forward(input_ids_2, attention_mask_2)
        # print(output_1)
        combined_output = torch.cat([output_1, output_2], dim=1)
        similarity_score = self.sts_head(combined_output).squeeze(1)
        return similarity_score
        # ### TODO
        # raise NotImplementedError

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
    
    def multiple_negatives_ranking_loss(self, embeddings1, embeddings2):
        """
        Implementation of MultipleNegativesRankingLoss.
        """

        # Compute cosine similarities between embeddings1 and all embeddings2
        scores = torch.matmul(embeddings1, embeddings2.T)  # (batch_size, batch_size)

        # Apply log-softmax to the scores
        log_probs = F.log_softmax(scores, dim=1)

        # The loss is the negative log likelihood of the correct (diagonal) elements
        loss = -torch.mean(torch.diag(log_probs))

        return loss
    
    def combined_loss(self, embeddings1, embeddings2, logits, targets):
        # Cosine Embedding Loss
        cosine_labels = 2 * targets.float() - 1  # Convert 0/1 to -1/1
        cosine_loss = F.cosine_embedding_loss(embeddings1, embeddings2, cosine_labels)
        
        # Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        
        # Multiple Negatives Ranking Loss
        ranking_loss = self.multiple_negatives_ranking_loss(embeddings1, embeddings2)
        
        # Combined Loss
        combined_loss = (
            self.weight_cosine * cosine_loss +
            self.weight_bce * bce_loss +
            self.weight_ranking * ranking_loss
        )
        
        return combined_loss
        
    def smart_regularized_loss(self, embeddings1, embeddings2, logits, targets):
        combined_loss = self.combined_loss(embeddings1, embeddings2, logits, targets)
        smart_loss = self.smart_loss(torch.cat((embeddings1, embeddings2), dim=1), logits)
        total_loss = combined_loss + smart_loss
        return total_loss
    
    
    def freeze_all_layers(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        self.frozen_layers = [True] * self.num_layers

    def unfreeze_all_layers(self):
        for param in self.bert.parameters():
            param.requires_grad = True
        self.frozen_layers = [False] * self.num_layers

    def freeze_layer(self, layer_num):
        if 0 <= layer_num < self.num_layers:
            for param in self.bert.bert_layers[layer_num].parameters():
                param.requires_grad = False
            self.frozen_layers[layer_num] = True

    def unfreeze_layer(self, layer_num):
        if 0 <= layer_num < self.num_layers:
            for param in self.bert.bert_layers[layer_num].parameters():
                param.requires_grad = True
            self.frozen_layers[layer_num] = False

    def unfreeze_layers(self, num_layers):
        for i in range(self.num_layers - num_layers, self.num_layers):
            self.unfreeze_layer(i)
        
        # Always unfreeze the pooler
        for param in self.bert.pooler_dense.parameters():
            param.requires_grad = True

    def get_frozen_layers_count(self):
        return sum(self.frozen_layers)

    def print_frozen_status(self):
        for i, frozen in enumerate(self.frozen_layers):
            print(f"Layer {i}: {'Frozen' if frozen else 'Unfrozen'}")
        # print(f"Pooler: {'Frozen' if next(self.bert.pooler_dense.parameters()).requires_grad == False else 'Unfrozen'}")
        print(f"Pooler: {'Frozen' if not any(param.requires_grad for name, param in self.bert.named_parameters() if 'pooler' in name) else 'Unfrozen'}")


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
    

class GradualUnfreezeScheduler(_LRScheduler):
    def __init__(self, optimizer, model, total_epochs, freeze_epochs=0, thaw_epochs=0, initial_lr=1e-5, max_lr=2e-5, final_lr=1e-5, last_epoch=-1):
        self.model = model
        self.total_epochs = total_epochs
        self.freeze_epochs = freeze_epochs
        self.thaw_epochs = thaw_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.num_layers = model.num_layers
        super(GradualUnfreezeScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        epoch = self.last_epoch + 1
        
        # Freezing phase
        if epoch <= self.freeze_epochs:
            self.model.freeze_all_layers()
            return [self.initial_lr for _ in self.base_lrs]
        
        # Thawing phase
        elif epoch <= self.freeze_epochs + self.thaw_epochs:
            thaw_progress = (epoch - self.freeze_epochs) / self.thaw_epochs
            layers_to_unfreeze = int(thaw_progress * self.num_layers)
            self.model.unfreeze_layers(layers_to_unfreeze)
            
            lr_progress = thaw_progress * (self.max_lr - self.initial_lr) / self.initial_lr
            return [self.initial_lr * (1 + lr_progress) for _ in self.base_lrs]
        
        # Fine-tuning phase
        else:
            self.model.unfreeze_all_layers()
            remaining_epochs = self.total_epochs - (self.freeze_epochs + self.thaw_epochs)
            fine_tune_progress = (epoch - (self.freeze_epochs + self.thaw_epochs)) / remaining_epochs
            lr_decay = (self.max_lr - self.final_lr) * fine_tune_progress
            return [self.max_lr - lr_decay for _ in self.base_lrs]

    def step(self):
        super().step()
        self.model.print_frozen_status()
        
    
class EarlyStopping:
    def __init__(self, patience=3, verbose=True, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time dev accuracy improved.
                            Default: 3
            verbose (bool): If True, prints a message for each improvement. 
                            Default: False
            delta (float): Minimum change in the monitored accuracy to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_epoch = 0
        self.path = path
        self.dev_acc_max = float("-inf")

    def __call__(self, dev_acc, model, epoch, optimizer, args, config):
        if self.best_score is None:
            self.best_score = dev_acc
            self.save_checkpoint(dev_acc, model, optimizer, args, config)
        elif dev_acc < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = dev_acc
            self.save_checkpoint(dev_acc, model, optimizer, args, config)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, dev_acc, model, optimizer, args, config):
        '''Saves model when dev accuracy increases.'''
        if self.verbose:
            print(f'Dev accuracy increased ({self.dev_acc_max:.6f} --> {dev_acc:.6f}).  Saving model ...')
        save_model(model, optimizer, args, config, self.path)
        self.dev_acc_max = dev_acc


# TODO Currently only trains on SST dataset!
def train_multitask(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load data
    # Create the data and its corresponding datasets and dataloader:
    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train"
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None

    # SST dataset
    if args.task == "sst" or args.task == "multitask":
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
        
    # Quora dataset (Paraphrase Detection)
    if args.task == "qqp" or args.task == "multitask":
        quora_train_data = SentencePairDataset(quora_train_data, args)
        quora_dev_data = SentencePairDataset(quora_dev_data, args)
        
        # subset_percentage = 10
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
    if args.task == "sts" or args.task == "multitask":
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
    optimizer_name = args.optimizer
    optimizer = get_optimizer(optimizer_name, params=model.parameters(), lr=lr)
    
    if args.scheduler == "onecycle":
        # Define the OneCycleLR scheduler
        max_lr = args.lr * 3
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(quora_train_dataloader), epochs=args.epochs)
    elif args.scheduler == "gradual_unfreeze":
        scheduler = GradualUnfreezeScheduler(
            optimizer, model, 
            total_epochs=args.epochs, 
            freeze_epochs=args.freeze_epochs, 
            thaw_epochs=args.thaw_epochs,
            initial_lr=args.initial_lr,
            max_lr=args.max_lr,
            final_lr=args.final_lr
        )
    
    
    # Setup EarlyStopping
    early_stopping = EarlyStopping(patience=3, verbose=True, path=args.filepath)
    
    best_dev_acc = float("-inf")

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        if args.task == "sst" or args.task == "multitask":
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
                loss = F.cross_entropy(logits, b_labels.view(-1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "sts" or args.task == "multitask":
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
                loss = F.mse_loss(logits, labels.float())
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
                
                embeddings1 = model.get_sentence_embeddings(b_ids_1, b_mask_1)
                embeddings2 = model.get_sentence_embeddings(b_ids_2, b_mask_2)
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                
                loss = model.smart_regularized_loss(embeddings1, embeddings2, logits, b_labels)
                
                loss.backward()
                optimizer.step()
                if args.scheduler == "onecycle":
                    scheduler.step()
                
                train_loss += loss.item()
                num_batches += 1
            
        if args.task == "etpc" or args.task == "multitask":
            # Trains the model on the etpc dataset
            ### TODO
            raise NotImplementedError

        train_loss = train_loss / num_batches

        quora_train_acc, _, _, sst_train_acc, _, _, sts_train_corr, _, _, etpc_train_acc, _, _ = (
            model_eval_multitask(
                sst_train_dataloader,
                quora_train_dataloader,
                sts_train_dataloader,
                etpc_train_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        quora_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_corr, _, _, etpc_dev_acc, _, _ = (
            model_eval_multitask(
                sst_dev_dataloader,
                quora_dev_dataloader,
                sts_dev_dataloader,
                etpc_dev_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        train_acc, dev_acc = {
            "sst": (sst_train_acc, sst_dev_acc),
            "sts": (sts_train_corr, sts_dev_corr),
            "qqp": (quora_train_acc, quora_dev_acc),
            "etpc": (etpc_train_acc, etpc_dev_acc),
            "multitask": (0, 0),  # TODO
        }[args.task]
        
        if args.scheduler=="gradual_unfreeze":
            # Move scheduler step to the end of the epoch for gradual unfreezing
            scheduler.step()
            print(f"Epoch {epoch+1} completed. Current learning rate: {scheduler.get_last_lr()[0]}")
            print(f"Frozen layers: {model.get_frozen_layers_count()} / {model.num_layers}")

        print(
            f"Epoch {epoch+1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
            print(f"New best model saved with dev accuracy: {best_dev_acc:.3f}")
        
        # Check early stopping criteria
        early_stopping(dev_acc, model, epoch, optimizer, args, config)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    # After training loop ends, ensure the best model according to EarlyStopping is loaded and saved
    if early_stopping.best_score > best_dev_acc:
        print(f"Early stopping model has better dev accuracy: {early_stopping.best_score:.3f} vs {best_dev_acc:.3f}")
        model.load_state_dict(torch.load(early_stopping.path)["model_config"])
        save_model(model, optimizer, args, config, args.filepath)
        print(f"Best model according to EarlyStopping reloaded and saved with dev accuracy: {early_stopping.best_score:.3f}")
    else:
        print(f"Best model already saved with dev accuracy: {best_dev_acc:.3f}")



def test_model(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        return test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()

    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='choose between "sst","sts","qqp","etpc","multitask" to train for different tasks ',
        choices=("sst", "sts", "qqp", "etpc", "multitask"),
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

    
    parser.add_argument(
        "--optimizer",
        type=str,
        help='choose between "adam" and "sophia"',
        choices=("adam", "sophia"),
        default="sophia",
    )
    
    parser.add_argument(
        "--pooling_type",
        type=str,
        choices=["cls", "mean", "max", "self_attention"],
        default="cls",
        help="Type of pooling to use for sentence embeddings"
    )
    
    parser.add_argument("--use_mha", action="store_true", help="Use Multi-Head Attention in the Paraphrase Classifier")
    parser.add_argument("--use_mlp", action="store_true", help="Use MLP Head in the Paraphrase Classifier")
    
    parser.add_argument("--scheduler", type=str, choices=["onecycle", "gradual_unfreeze", "None"], default="onecycle")
    parser.add_argument("--freeze_epochs", type=int, default=1)
    parser.add_argument("--thaw_epochs", type=int, default=2)
    parser.add_argument("--initial_lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=3e-5)
    parser.add_argument("--final_lr", type=float, default=1e-5)
    
    
    
    # Hyperparameters
    parser.add_argument("--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    
    parser.add_argument("--smart_lambda", type=float, default=0.02, help="Lambda for SMART regularization")
    
    parser.add_argument("--local_files_only", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"models/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)