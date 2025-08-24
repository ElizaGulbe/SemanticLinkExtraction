# Hypernymy and Synonymy Detection with Machine Learning

This repository contains a solution for detecting **hypernymy and synonymy** relationships using machine learning. The approach is designed to work in conjunction with the **Tēzaurs Database** and requires specific setup steps for connection and configuration.

## Prerequisites

To run this solution, you need:
- An established connection to the **Tēzaurs DB** (either the production DB or a copy). While the data exports from tezaurs DB are available on CLARIN e.g. https://repository.clarin.lv/repository/xmlui/handle/20.500.12574/119, the postgresql DB copies are available only on request, if you are interested in obtaining a copy of Tēzaurs DB, please contact Pēteris Paikens at @peteris@ailab.lv
- A custom `.env` file in the root directory with the following parameters:

  ```plaintext
  DB_HOST=localhost        # Your database hostname
  DB_PORT=5432             # Your database port
  DB_USER=username         # Your database username
  DB_PASSWORD=password     # Your database password
  DB_NAME=database_name    # Name of the database
  ```
In case the variables are not provided, default values (except for the password) will be used.
- python3 venv with pytorch, psycopg2, pandas, transformers, tqdm, dotenv libraries

## Link Detection Process

The link detection workflow is structured in three main stages : training process, candidate generation and link extraction. 

### 1. Training Process

The training process includes five key steps:

1. **Extract Positive Samples**: Establish a connection to Tēzaurs DB and extract existing semantic relations to use as positive samples for training data (1_Training process/Dataset/1_positive_example_extraction.py).
2. **Data Embedding** (compute-intensive, took 2h on MBP, compressed output is ~0.3GB): Convert textual data into embeddings using **HPLT embedding**. (1_Training process/Dataset/2_embed_positive_examples.py)
3. **Negative Sample Generation**: Generate negative examples for training purposes based on the positive examples. (1_Training process/Dataset/3_generate_training_dataset.py)
4. **Model Training**: Train the machine learning model on labeled data. (1_Training process/Model/model.py)
5. **Results Analysis**: Analyze model performance based on training results. (1_Training process/Model/Analysis tools)

#### Database Connection and Positive example extraction 

- The script for extracting positive examples is located at:

  ```plaintext
  1_Training process/Dataset/1_positive_example_extraction.py
  
The resulting csv file at `1_Training process/Dataset/1_positive_examples_nouns.csv` will contain all **unique** relationships recorded in the Latvian WordNet dataset. 

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
```


#### Negative Dataset Generation 

The negative examples are retrieved from the positive examples dataset that we obtained in the initial phase from Tēzaurs DB. 
To add negative examples to your dataset, run this file :
```plaintext
  1_Training process/Dataset/3_generate_training_dataset.py
```

During the research phase, we have discovered several different strategies for building the training dataset, speficically negative training data. 

1. Higher-level hypernyms: those not falling under direct hypernyms;
2. Random negative examples
3. Embedding vectors that have a close Euclidian distance but do not have an existing recorded relationship in the dataset
4. Unrelated senses of related words.
5. Similar/also/antonyms/holonyms - already recorded examples from different relation types that do not fall under the definition of hypernym or synonym 

#### Model training

The aim of the model training process is to learn to detect synonymy, hypernymy and none (other types that are not synonymy or hypernymy) relationship between two word senses.  

Input layer - a concatenation of word_1 embedding, sense_1 embedding, word_2 embedding, sense_2 embedding of size 3072;

Afterwards we implement series of linear and activation functions (you can change this in the search_space variable)

For the last layer we use SoftMax activation to convert raw output scores — into probabilities 

The output layer is a vecotr of size (3,) with displaying the probability of the relationship type (hypernymy, none, synonymy)

You can access the model training code here : 

```plaintext
    1_Training process/Model/model.py
```

We are using [ray library](https://docs.ray.io/en/latest/index.html) for the training process.  This library allows to test different model architectures, dropout rates, activation functions, optimizers and hyperparameter such as learning rate and epochs. Using this library you can report different metrics, checkpoints, weights etc, it is quite flexible.

**However**, I have noticed that sometimes this library experiences issues with the length of filepaths. When you load the the dataset, you must use the absolute path, not relative path. For some reason I also had problems with creating the reports of runs, so I saved them under the root directory and then copied the results into  
```plaintext
1_Training process/Previous results
```
#### Results analysis
You can access the results of existing runs here: 
```plaintext
1_Training process/Previous results
```
**TO DO** : pielikt atlikusos testus. 

To assess performance of different architectures, hyperparameter strategies, I've built 2 custom tools. 
1. A Dashboard that shows how different parameters in the search_space from the training process influence test dataset resuts located here 
```plaintext
1_Training process/Model/Analysis tools/dashboard_analysis.py
```
2. An excel file overview generator that displays each row as a training result instance and shows path to the weights located here : 
```plaintext
1_Training process/Model/Analysis tools/excel_overview.py
```
### 2. Candidate Generation

**Important** if you are using embeddings for semantic detection tasks in Latvian, I don't recommend considering (? missing ?)

Once we have trained the model to predict hypernymy, synonymy or other relationships, we can now apply the model to expansion of the WordNet. During the experimentation phase we mainly considered 3 strategies for synonymy or hypernymy candidate generation : 
1. For every sense in the Tēzaurs DB, we compare it with all other senses in the DB to find the highest probabilities for synonymy/hypernymy. We didn't opt for this strategy due to the fact that it is computationally expensive. 
2. Apply semantic relation rule-based extraction principles for nouns as described in [existing research](http://www.semti-kamols.lv/doc_upl/Kamols-Liepaja-raksts.pdf) from the definition of the word. (For hypernymy, however, you can also find patterns there, for synonymy, antonymy etc.)
3. Map Tēzaurs DB to exisitng synonym dictionary.(For synonymy)

This repository provides implementation code for the second and third strategy.  
### 3. Link extraction





