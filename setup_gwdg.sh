#!/bin/bash
set -e

# Set up Conda, install Python
module load anaconda3
conda create -n dnlp python=3.10
source activate dnlp

# Install packages
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tqdm==4.66.2 requests==2.31.0 transformers==4.38.2 tensorboard==2.16.2 tokenizers==0.15.1 -c conda-forge -c huggingface
pip install explainaboard-client==0.1.4 sacrebleu==2.4.0

# Download model on login-node
python -c "from tokenizer import BertTokenizer; from bert import BertModel; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('facebook/bart-base'); from transformers import BartModel; BartModel.from_pretrained('facebook/bart-base')"