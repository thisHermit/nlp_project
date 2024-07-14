#!/usr/bin/env python3

"""
This module contains our Dataset classes and functions to load the 3 datasets we're using.

You should only need to call load_multitask_data to get the training and dev examples
to train your model.
"""


import csv

import torch
from torch.utils.data import Dataset

from tokenizer import BertTokenizer


def preprocess_string(s):
    return " ".join(
        s.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sents": sents,
            "sent_ids": sent_ids,
        }

        return batched_data


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "sents": sents,
            "sent_ids": sent_ids,
        }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors="pt", padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors="pt", padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])
        token_type_ids = torch.LongTensor(encoding1["token_type_ids"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        ) = self.pad_data(all_data)

        batched_data = {
            "token_ids_1": token_ids,
            "token_type_ids_1": token_type_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "token_type_ids_2": token_type_ids2,
            "attention_mask_2": attention_mask2,
            "labels": labels,
            "sent_ids": sent_ids,
        }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors="pt", padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors="pt", padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])
        token_type_ids = torch.LongTensor(encoding1["token_type_ids"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])

        return (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            sent_ids,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            sent_ids,
        ) = self.pad_data(all_data)

        batched_data = {
            "token_ids_1": token_ids,
            "token_type_ids_1": token_type_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "token_type_ids_2": token_type_ids2,
            "attention_mask_2": attention_mask2,
            "sent_ids": sent_ids,
        }

        return batched_data


def load_multitask_data(sst_filename, quora_filename, sts_filename, etpc_filename, split="train"):
    sst_data = []
    num_labels = {}
    if split == "test":
        with open(sst_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent = record["sentence"].lower().strip()
                sent_id = record["id"].lower().strip()
                sst_data.append((sent, sent_id))
    else:
        with open(sst_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent = record["sentence"].lower().strip()
                sent_id = record["id"].lower().strip()
                label = int(record["sentiment"].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sst_data.append((sent, label, sent_id))

    print(f"Loaded {len(sst_data)} {split} examples from {sst_filename}")

    quora_data = []
    if split == "test":
        with open(quora_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                quora_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )

    else:
        with open(quora_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                try:
                    sent_id = record["id"].lower().strip()
                    quora_data.append(
                        (
                            preprocess_string(record["sentence1"]),
                            preprocess_string(record["sentence2"]),
                            int(float(record["is_duplicate"])),
                            sent_id,
                        )
                    )
                except:
                    pass

    print(f"Loaded {len(quora_data)} {split} examples from {quora_filename}")

    sts_data = []
    if split == "test":
        with open(sts_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                sts_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )
    else:
        with open(sts_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                sts_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        float(record["similarity"]),
                        sent_id,
                    )
                )

    print(f"Loaded {len(sts_data)} {split} examples from {sts_filename}")

    etpc_data = []
    if split == "test":
        with open(etpc_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                etpc_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )

    else:
        with open(etpc_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                try:
                    sent_id = record["id"].lower().strip()
                    etpc_data.append(
                        (
                            preprocess_string(record["sentence1"]),
                            preprocess_string(record["sentence2"]),
                            list(map(int, record["paraphrase_types"].strip("][").split(", "))),
                            sent_id,
                        )
                    )
                except:
                    pass

    print(f"Loaded {len(etpc_data)} {split} examples from {etpc_filename}")

    return sst_data, num_labels, quora_data, sts_data, etpc_data
