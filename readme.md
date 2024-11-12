# Hypernymy and Synonymy Detection with Machine Learning

This repository contains a solution for detecting hypernymy and synonymy relationships using machine learning. The approach is designed to work in conjunction with the **Tēzaurs Database** and requires specific setup steps for connection and configuration.

## Prerequisites

To run this solution, you need:
- An established connection to the **Tēzaurs DB** (either the production DB or a copy).
- A custom `.env` file in the root directory with the following parameters:

  ```plaintext
  DB_USER=username         # Your database username
  DB_PASSWORD=password     # Your database password
  DB_NAME=database_name    # Name of the database

## Link Detection Process

The link detection workflow is structured in three main stages:

### 1. Training Process

The training process includes five key steps:

1. **Database Connection**: Establish a connection to Tēzaurs DB.
2. **Data Embedding**: Convert textual data into embeddings using **HPLT embedding**.
3. **Negative Dataset Generation**: Generate negative examples for training purposes based on the positive examples.
4. **Model Training**: Train the machine learning model on labeled data.
5. **Results Analysis**: Analyze model performance based on training results.



*To-Do*: Include precise definitions for each relationship type.

#### Database Connection and Positive example extraction 

- The script for extracting positive examples is located at:

  ```plaintext
  1_Training process/Dataset/1_positive_example_extraction.py
  
The resulting csv file will generate all **unique** relationships recorded in the Latvian WordNet dataset. 

#### Database Connection and Positive example extraction 

- **Synonymy**
- **Hypernymy**
- **Antonymy**
- **Holonymy**
- **Similar**
- **Also**

#### Data embedding 

We embed the labeled data using **HPLT embedding** (see [HPLT embedding on Hugging Face](https://huggingface.co/HPLT/hplt_bert_base_lv)).

- The script for extracting positive examples is located at:
  ```plaintext
  1_Training process/Dataset/2_embed_positive_examples.py

#### Negative Dataset Generation
TO DO : High-level aprakstit visas pieejamas strategijas



