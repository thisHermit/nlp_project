# c[ ]

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-orange.svg)](https://pytorch.org/)
[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Final](https://img.shields.io/badge/Status-Final-purple.svg)](https://https://img.shields.io/badge/Status-Final-blue.svg)
[![AI-Usage Card](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](ai-usage-card.pdf)

- **Group name:** c[]
- **Group code:** G05
- **Group repository:** https://github.com/thisHermit/nlp_project
- **Tutor responsible:** Finn
- **Group team leader:** Madkour, Khaled
- **Group members:**
  - Madkour, Khaled
  - Khan, Bashar Jaan: basharjaankhan[at]gmail.com
  - Khan, Muneeb
  - Assy, Ahmed Tamer

# Setup instructions

> Explain how we can run your code in this section. We should be able to reproduce the results you've obtained.

> In addition, if you used libraries that were not included in the conda environment 'dnlp' explain the exact installation instructions or provide a `.sh` file for the installation.

> Which files do we have to execute to train/evaluate your models? Write down the command which you used to execute the experiments. We should be able to reproduce the experiments/results.

> _Hint_: At the end of the project you can set up a new environment and follow your setup instructions making sure they are sufficient and if you can reproduce your results.

> Following the setup instructions for the different tasks:

### Paraphrase Type Detection

#### Setup

This task uses only one external library `smart_pytorch`

```bash
pip install smart_pytorch
```

The model was trained and evaluated on the [Grete cluster provided by GWDG](https://gwdg.de/hpc/systems/grete/) on a single H100. To reproduce the experiments, the following command requests a H100 for 2 hours, which are sufficient to run each experiment independently.

```bash
srun -p grete-h100 --pty -n 1 -c 64 -G H100:1 --time 1:00:00 bash
```

<details>
<summary>Command to request resources for <code>&lt;hours&gt;</code> amount of time.</summary>

```bash
srun -p grete-h100 --pty -n 1 -c 64 -G H100:1 --time <hours>:00:00 bash
```

To run all experiments for Paraphrase Type Detection, 6 hours should be more than sufficient.

More details on the [srun options can be found here](https://slurm.schedmd.com/srun.html).

</details>

#### Reproducing experiments

To run any of the experiments, you switch to the corresponding experiment branch and then run the detection script `bart_detection.py`. The general command structure looks like:

```bash
git checkout ptd-exp<x> # here <x> is a branch number
conda activate dnlp # activate the conda environment
python3 bart_detection.py --use_gpu # run the experiment
```

<details>

<summary>bash commands to run each experiment</summary>

#### Experiment 1

```bash
git checkout ptd-exp2
conda activate dnlp
python3 bart_detection.py --use_gpu
```

#### Experiment 2

```bash
git checkout ptd-exp3
conda activate dnlp
python3 bart_detection.py --use_gpu
```

#### Experiment 3

```bash
git checkout ptd-exp4
conda activate dnlp
python3 bart_detection.py --use_gpu
```

#### Experiment 4

```bash
git checkout ptd-exp5
conda activate dnlp
python3 bart_detection.py --use_gpu
```

#### Experiment 5

```bash
git checkout ptd-exp6
conda activate dnlp
python3 bart_detection.py --use_gpu
```

#### Experiment 6

```bash
git checkout ptd-exp7
conda activate dnlp
python3 bart_detection.py --use_gpu
```

#### Experiment 7

```bash
git checkout ptd-exp8
conda activate dnlp
python3 bart_detection.py --use_gpu --optuna_optim
```

<details>
<summary><i>Off by one counter error for branch names</i></summary>
Please note that the branch named ptd-exp1 is actually the baseline model branch with the latest commits changes merged in and not the first experiment and so the count in the branch names is off by one.
</details>

</details>

### Paraphrase Type Generation

#### PAWS

Download dataset

```bash
cd data
wget -O train.parquet https://huggingface.co/datasets/google-research-datasets/paws/resolve/main/labeled_final/train-00000-of-00001.parquet
```

```
TOKENIZERS_PARALLELISM=true python3 bart_generation.py --use_gpu
```

Running this commands generates a checkpoint file which saves the model after pre-training on the paws dataset. The model is saved in the file `paws_bart_generation_model.ckpt`.

# Methodology

> In this section explain what and how you did your project.

> If you are unsure how this is done, check any research paper. They all describe their methods/processes. Describe briefly the ideas that you implemented to improve the model. Make sure to indicate how are you using existing ideas and extending them. We should be able to understand your project's contribution.

The methodology of all the different tasks is given below.

## Smart Loss

We implemented a "smart loss" function designed to weigh the importance of harder-to-classify examples during training. This approach aimed to mitigate the impact of class imbalance by dynamically adjusting the loss contribution based on the model's confidence in each prediction. By penalizing incorrect classifications more heavily, especially for underrepresented classes, we improved the model's ability to distinguish between paraphrase types.

## VAE

We introduced a Variational Autoencoder (VAE) into our paraphrase type detection pipeline to capture the latent distributions of paraphrase types more effectively. The VAE component allowed the model to generate paraphrase representations that maintain both diversity and coherence, leading to better generalization across different paraphrase types.

![mixed effects model](images/bart_vae.drawio.png)

There are random effects that come from the decoder part of the VAE and fixed effects that directly come from the BART model. This combination allowed us to generate more nuanced paraphrase embeddings, enhancing the model's performance on unseen data.

## Focal Loss

To further address the issue of class imbalance, we experimented with focal loss, which down-weights easy-to-classify examples and focuses on harder ones. This modification was particularly useful in scenarios where certain paraphrase types were underrepresented in the training data, helping the model to learn more effectively from these challenging cases.

## Simultaneous Training

We also explored simultaneous training on multiple datasets, including the Quora Question Pairs and other related datasets, to enhance the model's robustness. This approach allowed the model to leverage diverse paraphrase examples during training, resulting in a better understanding of paraphrase type variations across different contexts.

## Cosine Embedding Loss

We introduced Cosine Embedding Loss as a complementary objective to the traditional cross-entropy loss to enhance the semantic quality of the generated paraphrases. While cross-entropy loss focuses on ensuring that the generated output closely matches the target sequence at the token level, it does not explicitly guarantee that the meaning of the generated paraphrase aligns with the target. By incorporating Cosine Embedding Loss, we encouraged the model to generate outputs that are not only syntactically correct but also semantically similar to the target paraphrases.

In our scenario, we implemented Cosine Embedding Loss by extracting the embeddings of both the generated paraphrase and the target paraphrase from BART’s encoder. We then computed the cosine similarity between these embeddings, applying the loss to minimize the difference in meaning between the two sequences. This dual-loss approach allowed us to optimize the model for both grammatical accuracy and semantic fidelity, resulting in more meaningful and contextually appropriate paraphrases.

## Identity Loss

To further enhance the quality of the generated paraphrases and prevent the model from simply replicating the input sequence, we introduced an Identity Loss function. The primary goal of this loss was to discourage the model from generating outputs that are too similar to the input, thus promoting more diverse and meaningful paraphrasing. The Identity Loss was calculated by comparing the input and generated sequences, specifically checking if one text was a substring of the other. If the smaller text appeared within the larger text, a loss of 1 was incurred, otherwise, the loss was set to 0.

In our implementation, we decoded the input and generated sequences from their tokenized form back into text. We then compared the texts to determine if one was a substring of the other. This approach allowed us to penalize the model whenever it produced outputs that closely mirrored the input, thereby encouraging it to generate more varied paraphrases. This method helped to mitigate the risk of the model taking shortcuts by reproducing input sentences with minimal changes, which is often a concern in paraphrase generation tasks.

By combining these three losses during training, we created a multi-objective optimization process where the model was encouraged to generate paraphrases that are accurate (Cross Entropy Loss), semantically meaningful (Cosine Loss), and diverse (Identity Loss).

## Fine-Tuning BART on the PAWS Dataset for Paraphrase Type Generation

To enhance the performance of our paraphrase type generation model, we implemented a two-stage fine-tuning process using BART, starting with the PAWS dataset. The PAWS (Paraphrase Adversaries from Word Scrambling) dataset is a challenging dataset that contains pairs of sentences where word order has been swapped or word substitutions have been made, making it difficult for models to rely on surface form similarity to identify paraphrases. The dataset is designed to test and improve models' ability to capture deep semantic similarity, which aligns closely with the goals of our task.
Fine-Tuning on PAWS

### Objective:

The first step in our methodology was to fine-tune BART on the PAWS dataset, specifically using only the paraphrase pairs labeled as 1 (indicating that the pairs are paraphrases). The rationale behind this choice was that the PAWS dataset is relatively large and challenging, providing a robust starting point for paraphrase-related tasks. By fine-tuning on this dataset, we aimed to leverage the semantic complexity captured in PAWS to enhance BART's ability to generate high-quality paraphrases in our specific task.

### Procedure:

We extracted the paraphrase pairs from the PAWS dataset, filtering out non-paraphrase pairs to focus solely on those labeled as 1. The fine-tuning process involved training BART on these pairs, allowing the model to learn the nuances of paraphrasing from a dataset that closely mimics the kind of sentence structure and semantic similarity challenges present in our target task.

## Fine-Tuning on Our Dataset

After fine-tuning on PAWS, we used the resulting weights as the starting point for fine-tuning BART on our specific dataset for paraphrase type generation. This transfer learning approach was designed to further tailor the model's capabilities to our task, while leveraging the generalized paraphrasing abilities it acquired from the PAWS dataset.

# Experiments

> Keep track of your experiments here. What are the experiments? Which tasks and models are you considering?

> Write down all the main experiments and results you did, even if they didn't yield an improved performance. Bad results are also results. The main findings/trends should be discussed properly. Why a specific model was better/worse than the other?

> You are **required** to implement one baseline and improvement per task. Of course, you can include more experiments/improvements and discuss them.

> You are free to include other metrics in your evaluation to have a more complete discussion.

> Be creative and ambitious.

> For each experiment answer briefly the questions:

> - What experiments are you executing? Don't forget to tell how you are evaluating things.
> - What were your expectations for this experiment?
> - What have you changed compared to the base model (or to previous experiments, if you run experiments on top of each other)?
> - What were the results?
> - Add relevant metrics and plots that describe the outcome of the experiment well.
> - Discuss the results. Why did improvement _A_ perform better/worse compared to other improvements? Did the outcome match your expectations? Can you recognize any trends or patterns?

### Paraphrase Type Generation

All experiments for this task are evaulated using MCC.

#### Experiment 1

- What experiments are you executing? Don't forget to tell how you are evaluating things.
  - I use the decoder part of a VAE with skip connections to emulate a mixed effects model.
- What were your expectations for this experiment?
  - The latent parts of the vae model would capture some aspects of the classification space that would not be captured by a fully connected layer
- What have you changed compared to the base model (or to previous experiments, if you run experiments on top of each other)?
  - The decoder part of the vae is added on top of the base model.
- What were the results?
  - The model performed sligthtly better with a mcc of 0.066.
- Add relevant metrics and plots that describe the outcome of the experiment well.
  ![train accuracy](images/ptd-experiments/exp2.txt-metrics.csvtotal_train_accuracy_vs_epoch.png)![dev accuracy](images/ptd-experiments/exp2.txt-metrics.csvvalidation_accuracy_vs_epoch.png)

  ![train mcc](images/ptd-experiments/exp2.txt-metrics.csvtotal_train_matthews_coefficient_vs_epoch.png)![dev mcc](images/ptd-experiments/exp2.txt-metrics.csvvalidation_matthews_coefficient_vs_epoch.png)

- Discuss the results. Why did improvement _A_ perform better/worse compared to other improvements? Did the outcome match your expectations? Can you recognize any trends or patterns?

## Results

Summarize all the results of your experiments in tables:

| **Stanford Sentiment Treebank (SST)** | **Metric 1** | **Metric n** |
| ------------------------------------- | ------------ | ------------ |
| Baseline                              | 45.23%       | ...          |
| Improvement 1                         | 58.56%       | ...          |
| Improvement 2                         | 52.11%       | ...          |
| ...                                   | ...          | ...          |

| **Quora Question Pairs (QQP)** | **Metric 1** | **Metric n** |
| ------------------------------ | ------------ | ------------ |
| Baseline                       | 45.23%       | ...          |
| Improvement 1                  | 58.56%       | ...          |
| Improvement 2                  | 52.11%       | ...          |
| ...                            | ...          | ...          |

| **Semantic Textual Similarity (STS)** | **Metric 1** | **Metric n** |
| ------------------------------------- | ------------ | ------------ |
| Baseline                              | 45.23%       | ...          |
| Improvement 1                         | 58.56%       | ...          |
| Improvement 2                         | 52.11%       | ...          |
| ...                                   | ...          | ...          |

| **Paraphrase Type Detection (PTD)** | **Accuracy** | **MCC** |
| ----------------------------------- | ------------ | ------- |
| Baseline (exp1)                     | 79.2%\*      | 0.049   |
| VAE (exp3)                          | 80.9%        | 0.066   |
| VAE + smart loss (exp4)             | 79.8%        | 0.068   |
| Smart loss (exp5)                   | 79.6%        | 0.096   |
| Simultaneos training (exp6)         | 81.1%        | 0.081   |
| Deep layers with Focal Loss (exp7)  | 79.6%        | 0.062   |

| **Paraphrase Type Generation (PTG)** | **Metric 1** | **Metric n** |
| ------------------------------------ | ------------ | ------------ |
| Baseline                             | 45.23%       | ...          |
| Improvement 1                        | 58.56%       | ...          |
| Improvement 2                        | 52.11%       | ...          |
| ...                                  | ...          | ...          |

Notes:

- \*_These metrics for the baseline were observed after attempting to confirm and reproduce the final results. The initial recording of these metrics were 82.1% accuracy and 0.069 mcc._

> Discuss your results, observations, correlations, etc.

> Results should have three-digit precision.

### Hyperparameter Optimization

> Describe briefly how you found your optimal hyperparameter. If you focussed strongly on Hyperparameter Optimization, you can also include it in the Experiment section.

> _Note: Random parameter optimization with no motivation/discussion is not interesting and will be graded accordingly_

## Visualizations

> Add relevant graphs of your experiments here. Those graphs should show relevant metrics (accuracy, validation loss, etc.) during the training. Compare the different training processes of your improvements in those graphs.

> For example, you could analyze different questions with those plots like:

> - Does improvement A converge faster during training than improvement B?
> - Does Improvement B converge slower but perform better in the end?
> - etc...

## Members Contribution

> Explain what member did what in the project:

**Madkour, Khaled:** _implemented the training objective using X, Y, and Z. Supported member 2 in refactoring the code. Data cleaning, etc._

**Khan, Bashar Jaan:** _implemented all the experiments for paraphrase type detection. Supported Muneeb for desinging experiments for paraphrase detection and for paraphrase type generation (identity and paws)._

**Khan, Muneeb:** _implemented all the experiments for paraphrase detection. Supported Bashar for desinging experiments for paraphrase type detection and for paraphrase type generation._

**Assy, Ahmed Tamer:** _implemented the training objective using X, Y, and Z. Supported member 2 in refactoring the code. Data cleaning, etc._

# AI-Usage Card

> Artificial Intelligence (AI) aided the development of this project. Please add a link to your AI-Usage card [here](https://ai-cards.org/).

# References

Write down all your references (other repositories, papers, etc.) that you used for your project.

### TODO: convert links below to proper references

- https://arxiv.org/abs/1312.6114
- https://arxiv.org/abs/1708.02002
- Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Tuo Zhao. SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2177–2190, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.197.
