# DNLP SS24 Final Project

This is the starting code for the default final project for the Deep Learning for Natural Language Processing course at the University of Göttingen. You can find the handout [here](https://docs.google.com/document/d/1pZiPDbcUVhU9ODeMUI_lXZKQWSsxr7GO/edit?usp=sharing&ouid=112211987267179322743&rtpof=true&sd=true)

In this project, you will implement some important components of the BERT model to better understanding its architecture.
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh` (Use `setup_gwdg.sh` if you are using the GWDG clusters).
* Libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).
* Use this template to create your README file of your repository: <https://github.com/gipplab/dnlp_readme_template>

## Project Description

Please refer to the project description for a through explanation of the project and its parts.

### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

For the 2024 edition of the DNLP course at the University of Göttingen, the project was modified by [Niklas Bauer](https://github.com/ItsNiklas/), [Jonas Lührs](https://github.com/JonasLuehrs), ...
