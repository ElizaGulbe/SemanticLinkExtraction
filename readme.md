# Hypernymy and Synonymy Detection with Machine Learning

This repository contains a solution for detecting **hypernymy and synonymy** relationships using machine learning. The approach is designed to work in conjunction with the **Tēzaurs Database** and requires specific setup steps for connection and configuration.

## Prerequisites

To run this solution, you need:
- An established connection to the **Tēzaurs DB** (either the production DB or a copy). If you are interested in obtaining a copy of Tēzaurs DB, please contact Pēteris Paikens at @peterisp@gmail.com
- A custom `.env` file in the root directory with the following parameters:

  ```plaintext
  DB_USER=username         # Your database username
  DB_PASSWORD=password     # Your database password
  DB_NAME=database_name    # Name of the database


You can access the database by By contacting peteris@ailab.lv
## Link Detection Process

The link detection workflow is structured in three main stages : training process, candidate generatin and link extraction. 

### 1. Training Process

The training process includes five key steps:

1. **Database Connection**: Establish a connection to Tēzaurs DB. (1_Training process/Dataset/1_positive_example_extraction.py)
2. **Data Embedding**: Convert textual data into embeddings using **HPLT embedding**. (1_Training process/Dataset/2_embed_positive_examples.py)
3. **Negative Sample Generation**: Generate negative examples for training purposes based on the positive examples. (1_Training process/Dataset/3_generate_training_dataset.py)
4. **Model Training**: Train the machine learning model on labeled data. (1_Training process/Model/model.py)
5. **Results Analysis**: Analyze model performance based on training results. (1_Training process/Model/Analysis tools)

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

If you want to read the definitions of each of the semantic link types, please visit this resource: 
https://wordnet.ailab.lv/data/documents/WordNet_vadlīnijas.pdf   - Section 2.1 (In Latvian)

**Because of the quantity of the examples, we only train on synonymy and hypernymy examples.**


#### Data embedding 

We embed the labeled data using **HPLT embedding** (see [HPLT embedding on Hugging Face](https://huggingface.co/HPLT/hplt_bert_base_lv)).

- The script for extracting positive examples is located at:
  ```plaintext
  1_Training process/Dataset/2_embed_positive_examples.py


#### Negative Dataset Generation 

The negative examples are retrieved from the positive examples dataset that we obtained in the initial phase from Tēzaurs DB. 
To add negative examples to your dataset, run this file :
  ```plaintext
  1_Training process/Dataset/3_generate_training_dataset.py

During the research phase, we have discovered several different strategies for building the training dataset, speficically negative training data. 

1. Higher-level hypernyms: those not falling under direct hypernyms;
2. Random negative examples
3. Embedding vectors that have a close Eucledian distance but do not have an existing recorded relationship in the dataset
4. Unrelated senses of related words.
5. Similar/also/antonyms/holonyms - already recorded examples from different relation types that do not fall under the definition of hypernym or synonym 

#### Model training

You can access the model training code here  :

  ```plaintext
  1_Training process/Model/model.py

We are using ray library - https://docs.ray.io/en/latest/index.html for the training process.  This library allows to test different model architectures, dropout rates, activation functions, optimizers and hyperparameter such as learning rate and epochs. Using this library you can report different metrics, checkpoints, weights etc, it is quite flexible.

**However**, I have noticed that sometimes this library experiences issues with the length of filepaths. When you load the the dataset, you must use the absolute path, not relative path. For some reason I also had problems with creating the reports of runs, so I saved them under the root directory. 

#### Results analysis
TO DO : High-level aprakstit visas pieejamas strategijas

### 2. Candidate Generation
### 3. Link extraction





