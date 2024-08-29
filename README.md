# Detecting AI reviews using deep learning architectures

## Authors
[Amzi Jeffs](https://github.com/AmziJeffs)    
[Junichi Koganemaru](https://github.com/jkoganem)  
[Salil Singh](https://github.com/sllsnghlrns)  
[Ashwin Tarikere](https://github.com/ashwintan1)     


## Overview

With the explosion of LLMs and NLP methods, AI-generated text has become ubiquitous on the internet. This presents several challenges across many contexts, ranging from plagiarism in the academic setting to misinformation on social media and its consequences in electoral politics. With this in mind, we explore a range of classical statistical learning classifiers as well as deep learning based transformers to detect AI-generated text. 

## Structure of repository

- `raw data` contains a readme citing our data sources, and our exploratory data analysis notebooks. It does not contain the raw data itself, which was too large to add to the repository. Instead, the `processing.ipynb` notebook can be used to reconstruct our dataset from the source datasets that we link.
- `code` contains our Python scripts for XGBoost and fine-tuning, which are intended to run on a powerful computing cluster.

## Our Dataset: 

Consisted of 10000 human-generated and 10000 AI-generated text snippets sourced from various contexts and models. 

- Product Reviews generated by GPT-2 from a [Kaggle competition](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset
).
- Essays generated by a range of models, also from a [Kaggle competition](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
).
- News articles generated by [Grover](https://github.com/rowanz/grover/tree/master
).
- Wikipedia intros generated by GPT-3 Curie, from a Hugging Face [repo](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro
). 

 
## Evaluation metric



## Acknowledgements 

We would like to thank the Erdös institute for providing the authors the opportunity to work on this project as part of the Erdös institute Deep Learning Bootcamp. 

We would also like to thank Nuno Chagas and the Department of Mathematical Sciences at Carnegie Mellon University for providing computing support for the project. 



