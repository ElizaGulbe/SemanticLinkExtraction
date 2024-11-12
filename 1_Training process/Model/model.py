import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import pandas as pd
import pickle
from tqdm import tqdm 
from ray import tune, train
from ray.tune import grid_search
import ray
import os
import tempfile
from ray.train import Checkpoint
from torch.utils.data import DataLoader, TensorDataset
def deserialize_tensor(serialized_tensor):
        return pickle.loads(bytes.fromhex(serialized_tensor))


class DynamicSemanticRelationModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes, activation, dropout_rate):
        super(DynamicSemanticRelationModel, self).__init__()
        
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(p=dropout_rate))  # Add dropout after each activation layer
            in_size = hidden_size
            
        layers.append(nn.Linear(in_size, num_classes))
        
        self.model = nn.Sequential(*layers)
        # Add Softmax layer
        self.softmax = nn.Softmax(dim=1)  # Apply softmax to the last dimension
    
    def forward(self, x):
        x = self.model(x)
        return self.softmax(x) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(percentage_of_postive):
    df = pd.read_csv(r"C:\Users\davis\OneDrive\Desktop\Eliza\ai.lab_programming\Production\training_dataset_nouns_with_none_hypernyms_synonyms_heading_new2910.csv",usecols=['sense1_gloss_embedding', "sense2_gloss_embedding", "sense1_heading_embedding", "sense2_heading_embedding" ,"rel_type"])
    print(df.columns)
    df = df[df["rel_type"] != 'hyponym']
    values_to_replace = ['holonym', 'meronym', 'antonym', 'also', 'similar',"hyponym"]
    df['rel_type'] = df['rel_type'].replace(values_to_replace, 'none')
    count_non_none = df[~df['rel_type'].str.contains("none")].shape[0]
    total_rows = round(count_non_none / percentage_of_postive)
    count_none_grandparents = df[df['rel_type'] == 'none_grandparents'].shape[0]
    number_of_random_rows = total_rows- count_none_grandparents - df[df['rel_type'] == 'none'].shape[0]
    none_random_df = df[df['rel_type'] == 'none_random']
    selected_random = none_random_df.head(round(number_of_random_rows*0.10))
    none_similarity_df = df[df['rel_type'] == 'none_similarity']
    selected_similarity = none_similarity_df.head(round(number_of_random_rows*0.48))
    rows_per_type = round(number_of_random_rows * 0.42 / 3)
    none_gloss_df = df[df['rel_type'].isin(['none_gloss_synonym', 'none_gloss_hypernym', 'none_gloss_hyponym'])]
    print(none_gloss_df["rel_type"].value_counts())
    # Sample rows equally from each rel_type
    synonym_sample = none_gloss_df[none_gloss_df['rel_type'] == 'none_gloss_synonym'].sample(rows_per_type, random_state=1)
    hypernym_sample = none_gloss_df[none_gloss_df['rel_type'] == 'none_gloss_hypernym'].sample(rows_per_type, random_state=1)
    hyponym_sample = none_gloss_df[none_gloss_df['rel_type'] == 'none_gloss_hyponym'].sample(rows_per_type, random_state=1)
    df = pd.concat([selected_similarity,selected_random,synonym_sample, hyponym_sample,hypernym_sample ,df[~df['rel_type'].isin(['none_random', 'none_similarity',"none_gloss_hypernym",'none_gloss_synonym','none_gloss_hyponym'])]])
    df["sense1_gloss_embedding"] = df["sense1_gloss_embedding"].apply(deserialize_tensor)
    df["sense2_gloss_embedding"] = df["sense2_gloss_embedding"].apply(deserialize_tensor) 
    df["sense1_heading_embedding"] = df["sense1_heading_embedding"].apply(deserialize_tensor)
    df["sense2_heading_embedding"] = df["sense2_heading_embedding"].apply(deserialize_tensor) 
    df['rel_type'] = df['rel_type'].apply(lambda x: 'none' if 'none' in x else x)
    return df

def train_model(config):

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = get_data(percentage_of_postive=config["postive_percentage"])

    # Ensure all embeddings are moved to the CPU before using Pandas
    concatenated_vectors = df.apply(
        lambda row: torch.cat((
            row['sense1_heading_embedding'].flatten().cpu(),
            row['sense1_gloss_embedding'].flatten().cpu(),
            row['sense2_heading_embedding'].flatten().cpu(),
            row['sense2_gloss_embedding'].flatten().cpu()
        )),
        axis=1
    )

    # Stack all concatenated vectors into a single tensor and move to device
    X = torch.stack(concatenated_vectors.tolist()).to(device)
    print(X.shape)
    
    df['rel_type'] = pd.Categorical(df['rel_type'])
    y = torch.tensor(df['rel_type'].cat.codes.values, dtype=torch.long).to(device)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    # No need to use .to(device) on the Dataset or DataLoader, instead, ensure data is on the correct device.
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    model = DynamicSemanticRelationModel(
        input_size=768*4,  # Assuming 768 is the size of each word embedding
        num_classes=len(df["rel_type"].unique()),    # Adjust to the actual number of classes
        hidden_sizes=config["hidden_sizes"],
        activation=config["activation"],
        dropout_rate=config["dropout_rate"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])

    num_epochs = config["num_epochs"]

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Mini-batch training within each epoch
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(loss)

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        test_outputs = model(X_test)
        _, predicted_classes = torch.max(test_outputs, 1)
        
        # Move the predictions and true labels to CPU for evaluation
        y_test_cpu = y_test.cpu()
        predicted_classes_cpu = predicted_classes.cpu()

        # Calculate metrics
        accuracy = accuracy_score(y_test_cpu, predicted_classes_cpu)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_cpu, predicted_classes_cpu, average='weighted')
        conf_matrix = confusion_matrix(y_test_cpu, predicted_classes_cpu)
        class_report = classification_report(y_test_cpu, predicted_classes_cpu, target_names=df['rel_type'].cat.categories)

        # Report metrics to Ray Tune
        report = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
        }

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        torch.save(
            model.state_dict(),
            os.path.join(temp_checkpoint_dir, f"weights.pth"),
        )
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        train.report(report, checkpoint=checkpoint)

    # Save the model after training

# Define the search space
search_space = {
    "lr": tune.loguniform(1e-4,5e-5),  # Replace loguniform with grid search
    "num_epochs": tune.choice([50,70,100,140]),
    "batch_size": tune.choice([16,32,64]),
    "hidden_sizes": tune.choice([
        [1024, 512, 256, 128, 64],
        [2048,1024, 512, 256, 128, 64],
        [512],
        [1024],
        [512, 256, 128, 64],
        [2048, 512, 128]
    ]),
    "activation": tune.choice(['relu']),
    "optimizer": tune.choice(['Adam',]),
    "dropout_rate": tune.choice([0.0]),
    "postive_percentage" : tune.grid_search([0.5])
}
def dynamic_trial_name_creator(trial):
    return f"run{trial.trial_id}"
ray.init(num_cpus=12, num_gpus=1)
analysis = tune.run(
    train_model, # Your training function
    config=search_space,
    trial_dirname_creator=dynamic_trial_name_creator,
    storage_path="C:/ray_results",
    resources_per_trial={"cpu":12, "gpu": 1},
    num_samples=1500
)