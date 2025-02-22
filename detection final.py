# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:35:45 2024

@author: Support
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import csv
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import networkx as nx
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Start timing the execution
start_time = time.time()

# Function to apply sign flipping to weights and biases for each client (attacker behavior)
def apply_sign_flipping(weights):
    # Multiply each weight by -1 to simulate an attack
    return [-1 * w for w in weights]

# Set the paths to the dataset folders
dataset_path = "path/to/your/dataset_folder"    # Replace the placeholder with the actual path to your dataset folder
test_data_path = "path/to/your/test_subset_folder"  # Replace the placeholder with the actual path to your test data folder

# Get the round number from user input
num_round = int(input("Enter the number of rounds for Federated Learning: "))

# Get the number of clients from user input
num_clients = int(input("Enter the number of clients: "))

# Get the percentage of attackers
percentage_attackers = int(input("Enter the percent of attackers: "))

# Get the attack probability
alpha = float(input("Enter the attack probability among 0.5, 0.7, and 1.0: "))

# Get the threshold for classification based on loss
detection_threshold = float(input("Enter the threshold for classification: "))

# Define the GAT model for graph learning
class GATModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel, self).__init__()
        # Graph Attention Network layer
        self.gat_conv = GATConv(in_channels, out_channels, heads=1, concat=True)
        # Global pooling to obtain graph-level representation
        self.global_pool = global_mean_pool
        # Linear layer to classify the graph representation
        self.classifier = nn.Linear(out_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat_conv(x, edge_index)  # Apply GAT layer to obtain node-level features
        x = self.global_pool(x, data.batch)  # Apply global pooling to get a graph-level embedding
        logits = self.classifier(x)  # Graph-level classification logits
        return logits  # Output as a probability between 0 and 1

# Initialize the GAT model
gat_model = GATModel(in_channels=6, out_channels=16)
gat_optimizer = optim.Adam(gat_model.parameters(),lr=0.0005)  # Adjusted learning rate for GAT model
gat_criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss with logits

# Initialize and load the GAT model (pre-trained on all benign clients)
gat_model = GATModel(in_channels=6, out_channels=16)
gat_model_path = "path/to/your/GAT_model.pth"   # Replace the placeholder with the actual path to your pre-trained GAT model
gat_model.load_state_dict(torch.load(gat_model_path))
gat_model.eval()



# Create CSV files for storing metrics
avg_train_acc_file = open(
    f"path/to/your/average_accuracies_folder/{detection_threshold}_threshold_{percentage_attackers}%_attackers_alpha_{alpha}_attackers_average_accuracies.csv", 
    "w", 
    newline=""
)   # Replace the placeholder with the actual path to your average accuracies folder
avg_train_acc_writer = csv.writer(avg_train_acc_file)
avg_train_acc_writer.writerow(["Round", "Average Train Accuracy", "Average Loss"])

test_acc_file = open(
    f"path/to/your/average_accuracies_folder/{detection_threshold}_threshold_{percentage_attackers}%_attackers_alpha_{alpha}_attackers_test_accuracies.csv", 
    "w", 
    newline=""
)   # Replace the placeholder with the actual path to your average accuracies folder
test_acc_writer = csv.writer(test_acc_file)
test_acc_writer.writerow(["Round", "Test Accuracy"])

csv_file = open(
    f"path/to/your/GAT_detection_folder/{detection_threshold}_threshold_{percentage_attackers}%_attackers_alpha_{alpha}_attackers_client_info.csv", 
    "w", 
    newline=""
)   # Replace the placeholder with the actual path to your GAT detection folder
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Scenario", "Round", "Client Number", "Client Name", "Num Samples", "Train Accuracies", "Train Loss"])

evaluation_metrics_file = open(
    f"path/to/your/evaluation_metrics_folder/{detection_threshold}_threshold_{percentage_attackers}%_attackers_alpha_{alpha}_attackers_evaluation_metrics.csv", 
    "w", 
    newline=""
)   # Replace the placeholder with the actual path to your evaluation metrics folder
evaluation_metrics_writer = csv.writer(evaluation_metrics_file)
evaluation_metrics_writer.writerow(["Round", "Precision", "Recall", "F1 Score"])

attackers_csv_file_name = f"path/to/your/GAT_detection_folder/{detection_threshold}_threshold_{percentage_attackers}%_attackers_alpha_{alpha}_initial_attackers.csv"    # Replace the placeholder with the actual path to your GAT detection folder
attackers_csv_file = open(attackers_csv_file_name, "w", newline="")
attackers_csv_writer = csv.writer(attackers_csv_file)
attackers_csv_writer.writerow(['Attacker Number', 'Client Folder'])

attackers_appearances_file_name = f"path/to/your/GAT_detection_folder/{detection_threshold}_threshold_{percentage_attackers}%_attackers_alpha_{alpha}_attackers_appearances.csv"    # Replace the placeholder with the actual path to your GAT detection folder
attackers_appearances_file = open(attackers_appearances_file_name, "w", newline="")
attackers_appearances_writer = csv.writer(attackers_appearances_file)
attackers_appearances_writer.writerow(['Round', 'Potential Attackers', 'Actual Attackers'])

# Create CSV files for GAT predictions of benign and attacker clients
gat_benign_prediction_file = open(
    f"path/to/your/GAT_metrics_folder/{detection_threshold}_threshold_{percentage_attackers}%_attackers_alpha_{alpha}_benign_gat_predictions.csv", 
    "w", 
    newline=""
)   # Replace the placeholder with the actual path to your GAT metrics folder
gat_benign_prediction_writer = csv.writer(gat_benign_prediction_file)
gat_benign_prediction_writer.writerow(["Round", "Client Number", "Client Folder", "Actual Label", "Predicted Label"])

gat_attacker_prediction_file = open(
    f"path/to/your/GAT_metrics_folder/{detection_threshold}_threshold_{percentage_attackers}%_attackers_alpha_{alpha}_attacker_gat_predictions.csv", 
    "w", 
    newline=""
)   # Replace the placeholder with the actual path to your GAT metrics folder
gat_attacker_prediction_writer = csv.writer(gat_attacker_prediction_file)
gat_attacker_prediction_writer.writerow(["Round", "Client Number", "Client Folder", "Actual Label", "Predicted Label"])

# Create CSV files for detection rate
detection_summary_file = open(
    f"path/to/your/GAT_metrics_folder/{detection_threshold}_threshold_{percentage_attackers}%_attackers_alpha_{alpha}_detection_summary.csv", 
    "w", 
    newline=""
)   # Replace the placeholder with the actual path to your GAT metrics folder
detection_summary_writer = csv.writer(detection_summary_file)
detection_summary_writer.writerow(["Round", "Total Attackers Appeared", "Detected Attackers", "Detection Rate"])

# Create CSV files for recording GAT metrics
gat_metrics_file = open(
    f"path/to/your/GAT_metrics_folder/{detection_threshold}_threshold_{percentage_attackers}%_attackers_alpha_{alpha}_gat_metrics.csv", 
    "w", 
    newline=""
)   # Replace the placeholder with the actual path to your GAT metrics folder
gat_metrics_writer = csv.writer(gat_metrics_file)
gat_metrics_writer.writerow(["Round", "TP", "FP", "TN", "FN", "FNR", "FPR", "Detection Rate", "False Detection Rate", "Test Accuracy", "Precision", "Recall", "F1 Score"])

# Initialize totals for overall metrics
TP_total = FP_total = TN_total = FN_total = 0

# Defining the global model
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Convolutional layer with 32 filters
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)  # Another convolutional layer with 32 filters
        self.pool2 = nn.MaxPool2d(2, 2)  # Second max pooling layer
        self.flatten = nn.Flatten()  # Flatten layer to convert 2D feature maps to 1D
        self.fc1 = nn.Linear(32 * 5 * 5, 1024)  # Fully connected layer with 1024 units
        self.fc2 = nn.Linear(1024, 62)  # Output layer with 62 units (number of classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Apply convolution, followed by ReLU activation
        x = self.pool(x)  # Apply max pooling
        x = torch.relu(self.conv2(x))  # Apply second convolution, followed by ReLU activation
        x = self.pool2(x)  # Apply second max pooling
        x = self.flatten(x)  # Flatten the feature maps
        x = torch.relu(self.fc1(x))  # Apply fully connected layer, followed by ReLU activation
        x = self.fc2(x)  # Output layer
        return x


# Initialize the global model
global_model = GlobalModel()

# Define the loss function and optimizer for the global model
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(global_model.parameters(),lr=0.001)  # Adam optimizer for global model training

# Get a list of client folders
all_client_folders = os.listdir(dataset_path)

# Initialize lists to store metrics
average_train_accuracies = []
average_train_losses = []
test_accuracies = []
precision_scores = []
recall_scores = []
f1_scores = []

# Initialize the attackers array
attackers = [False] * len(all_client_folders)

# Number of attackers (% of available clients)
num_attackers = int(len(all_client_folders) * (percentage_attackers / 100))

# Randomly select attackers and update the attackers array
attacker_indices = random.sample(range(len(all_client_folders)), num_attackers)
for i in attacker_indices:
    attackers[i] = True

# Split client folders into attackers and non-attackers
attacker_folders = [folder for i, folder in enumerate(all_client_folders) if attackers[i]]
non_attacker_folders = [folder for i, folder in enumerate(all_client_folders) if not attackers[i]]

# Set a seed value
random.seed(40)

# Initialize a counter for rounds with attacks
rounds_with_attacks = 0

# Main Federated Learning loop
for round_num in range(num_round):
    
    # Print the current round information
    print(f"\nRound {round_num + 1} - Percentage of Attackers: {percentage_attackers}%, Attack Probability: {alpha}")
    
    # Flag to track if an attack occurred in this round
    attack_occurred_this_round = False
    
    # Record the actual attacker
    actual_attackers = []
    detected_attackers = 0
    
    # Counter for TP,FP,TN,FN
    TP = FP = TN = FN = 0

    # Randomly select client folders from attacker and non-attacker pools
    num_attackers_in_selected = int(num_clients * (percentage_attackers / 100))
    selected_attacker_folders = random.sample(attacker_folders, num_attackers_in_selected)
    selected_non_attacker_folders = random.sample(non_attacker_folders, num_clients - num_attackers_in_selected)

    # Combine selected attacker and non-attacker folders
    selected_client_folders = selected_attacker_folders + selected_non_attacker_folders
    random.shuffle(selected_client_folders)  # Shuffle to make the selection random
    
    # Print the selected client folders
    print(f"\nSelected train client folders for round {round_num +1} and Percentage of Attackers: {percentage_attackers}%, Attack Probability: {alpha}:")
    for i, client_folder in enumerate(selected_client_folders):
        print(f"Client {i + 1}: {client_folder}")

    print(f"\nPotential attackers appearing in round {round_num + 1} and Percentage of Attackers: {percentage_attackers}%, Attack Probability: {alpha}:")
    potential_attackers = []  # List to store potential attackers
    for i, client_folder in enumerate(selected_client_folders):
        if client_folder in selected_attacker_folders:
            print(f"Potential Attacker {i + 1}: {client_folder}")
            potential_attackers.append(client_folder) 

    # Initialize global weights for aggregation
    global_weights = [param.data.clone() for param in global_model.parameters()]
    
    # Initialize a numpy array to store the number of samples for each client
    num_samples_per_client = np.zeros(len(selected_client_folders), dtype=int)
    
    # Initialize a list to store the training accuracies,loss of all clients in each round
    client_train_accuracies = []
    client_train_losses = []

    # Sequentially train clients
    for i, client_folder in enumerate(selected_client_folders):
        client_folder_path = os.path.join(dataset_path, client_folder, "train")
        
        # Prepare X_train and Y_train for the client
        X_train, Y_train = [], []
        num_samples = 0
        unique_classes = set()
        
        # Iterate over train folders within the client folder
        train_folders = os.listdir(client_folder_path)
        for train_folder in train_folders:
            train_folder_path = os.path.join(client_folder_path, train_folder)
            
            # Iterate over image files within the train folder
            image_files = os.listdir(train_folder_path)
            for image_file in image_files:
                image_path = os.path.join(train_folder_path, image_file)
                image = Image.open(image_path).convert('L')  # Convert image to grayscale
                transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])  # Resize and convert to tensor
                image_tensor = transform(image)
                X_train.append(image_tensor)
                Y_train.append(int(train_folder))  # Label the image based on folder name
                unique_classes.add(int(train_folder))
                num_samples += 1
        
        num_samples_per_client[i] = num_samples
        print(f"\nTraining on Client {i + 1}: {client_folder} - Number of Samples: {num_samples} - Number of Classes: {len(unique_classes)} for FL round {round_num + 1} and Percentage of Attackers: {percentage_attackers}%, Attack Probability: {alpha}")
        
        # Convert collected training data to DataLoader 
        X_train = torch.stack(X_train)
        Y_train = torch.tensor(Y_train)
        train_dataset = data.TensorDataset(X_train, Y_train)
        train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Initialize local model with global weights
        local_model = GlobalModel()
        local_model.load_state_dict(global_model.state_dict())
        local_optimizer = optim.Adam(local_model.parameters(), lr=0.001)

        # Train local model
        local_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_epochs = 20  # Number of epochs for training local model
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch + 1}/{num_epochs}')
            for inputs, labels in train_loader:
                local_optimizer.zero_grad()  # Reset gradients
                outputs = local_model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass to compute gradients
                local_optimizer.step()  # Update weights
                total_loss += loss.item() * inputs.size(0)  # Accumulate loss
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                correct += (predicted == labels).sum().item()  # Count correct predictions
                total += labels.size(0)  # Total number of samples

        train_accuracy = correct / total  # Calculate training accuracy
        train_loss = total_loss / total  # Calculate average training loss
        client_train_accuracies.append(train_accuracy)
        client_train_losses.append(train_loss)
        
        # Append performance metrics to CSV file
        csv_writer.writerow([f"{percentage_attackers}%_attackers_alpha_{alpha} attack", round_num + 1, i + 1, client_folder, num_samples, train_accuracy, train_loss])
        
        # Print training metrics for each client
        print(f"\nClient {i + 1} - Training Accuracy: {train_accuracy:.4f}, Training Loss: {train_loss:.4f}")
        
        # Save client weights before modification (benign scenario)
        before_modification_path = f"path/to/your/GAT_detection_folder/before_modification/{detection_threshold}_threshold_Attackers_{percentage_attackers}_alpha_{alpha}/Round_{round_num+1}_Client_{i+1}_{client_folder}.h5"    # Replace the placeholder with the actual path to your GAT detection "before_modification" folder    
        os.makedirs(os.path.dirname(before_modification_path), exist_ok=True)
        with h5py.File(before_modification_path, 'w') as file:
            for layer_idx, layer in enumerate(local_model.children()):
                if hasattr(layer, 'weight') and layer.weight is not None:
                    file.create_dataset(f'layer_{layer_idx}_weights', data=layer.weight.data.cpu().numpy())
                if hasattr(layer, 'bias') and layer.bias is not None:
                    file.create_dataset(f'layer_{layer_idx}_biases', data=layer.bias.data.cpu().numpy())
        
        # Determine if the client is an attacker and apply sign flipping if necessary
        if client_folder in selected_attacker_folders:
            # Generate x as random numbers from 1 to 100
            x = random.randint(1, 100)
            
            # Modify the weights and biases of the attacker
            if x <= alpha * 100:
                # An attack is launched in this round
                print(f"\nAttacker: {client_folder}, x: {x}, alpha: {alpha}")
                print(f"\nActual attack launched by Attacker: {client_folder} in round {round_num + 1} and Percentage of Attackers: {percentage_attackers}%, Attack Probability: {alpha}")
                
                # Track if an attack occurred in this round
                attack_occurred_this_round = True
                
                # Apply sign flipping to client weights to simulate attacker behavior
                client_weights = [param.data.clone() for param in local_model.parameters()]
                client_weights = apply_sign_flipping(client_weights)
                with torch.no_grad():
                    for param, flipped_weight in zip(local_model.parameters(), client_weights):
                        param.data.copy_(flipped_weight)
                        
                # Record the actual attacker
                actual_attackers.append(client_folder)
                
                # Save client weights after modification (attacker scenario)
                after_modification_path = f"path/to/your/GAT_detection_folder/after_modification/{detection_threshold}_threshold_Attackers_{percentage_attackers}_alpha_{alpha}/Round_{round_num+1}_ActualAttacker_{i+1}_{client_folder}.h5"
                os.makedirs(os.path.dirname(after_modification_path), exist_ok=True)
                with h5py.File(after_modification_path, 'w') as file:
                    for layer_idx, layer in enumerate(local_model.children()):
                        if hasattr(layer, 'weight') and layer.weight is not None:
                            file.create_dataset(f'layer_{layer_idx}_weights', data=layer.weight.data.cpu().numpy())
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            file.create_dataset(f'layer_{layer_idx}_biases', data=layer.bias.data.cpu().numpy())


        
        # Generate graph representation for the client's model
        G = nx.DiGraph()  # Initialize directed graph
        for idx, layer in enumerate(local_model.children()):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # Extract weights and biases for layers that have them
                weights = layer.weight.data.cpu().numpy()
                biases = layer.bias.data.cpu().numpy() if layer.bias is not None else None
                
                # Compute node features using absolute values of weights and biases
                weight_mean_abs = np.mean(np.abs(weights))
                weight_norm_abs = np.linalg.norm(np.abs(weights))
                weight_variance_abs = np.var(np.abs(weights))
                if biases is not None:
                    bias_mean_abs = np.mean(np.abs(biases))
                    bias_norm_abs = np.linalg.norm(np.abs(biases))
                    bias_variance_abs = np.var(np.abs(biases))
                else:
                    bias_mean_abs = bias_norm_abs = bias_variance_abs = 0.0
                
                
                # Add node to graph with computed features
                G.add_node(idx,
                           layer_name=layer.__class__.__name__,
                           weight_mean_abs=weight_mean_abs,
                           weight_norm_abs=weight_norm_abs,
                           weight_variance_abs=weight_variance_abs,
                           bias_mean_abs=bias_mean_abs,
                           bias_norm_abs=bias_norm_abs,
                           bias_variance_abs=bias_variance_abs)
            else:
                # Add node with dummy features for layers without weights or biases
                G.add_node(idx,
                           layer_name=layer.__class__.__name__,
                           weight_mean_abs=0.0,
                           weight_norm_abs=0.0,
                           weight_variance_abs=0.0,
                           bias_mean_abs=0.0,
                           bias_norm_abs=0.0,
                           bias_variance_abs=0.0)
            if idx > 0:
                G.add_edge(idx - 1, idx)  # Connect nodes sequentially

        # Prepare graph data for GAT model training
        features = torch.tensor([list(G.nodes[node].values())[1:] for node in G.nodes], dtype=torch.float)  # Extract node features
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()  # Extract edge list
        data_graph = Data(x=features, edge_index=edge_index)  # Create PyTorch Geometric Data object

        # Perform inference using the pre-trained GAT model
        with torch.no_grad():
            output = gat_model(data_graph).view(-1)
            target = torch.tensor([0.0 if client_folder not in actual_attackers else 1.0], dtype=torch.float)
            loss = gat_criterion(output, target)
            predicted_label = 0 if loss < detection_threshold else 1  # Classify based on loss threshold
            if predicted_label == 1 and client_folder in actual_attackers:
                detected_attackers += 1
                TP += 1
            elif predicted_label == 1 and client_folder not in actual_attackers:
                FP += 1
            elif predicted_label == 0 and client_folder not in actual_attackers:
                TN += 1
            elif predicted_label == 0 and client_folder in actual_attackers:
                FN += 1
            
            # Log GAT prediction separately for benign and attacker clients
            if client_folder in actual_attackers:
                label = 1  # Attacker
                gat_attacker_prediction_writer.writerow([round_num + 1, i + 1, client_folder, label, predicted_label])
            else:
                label = 0  # Benign
                gat_benign_prediction_writer.writerow([round_num + 1, i + 1, client_folder, label, predicted_label])
            
            # Print GAT inference result
            print(f"Round {round_num + 1}, Client {i + 1} - Loss: {loss.item():.4f}, Predicted Label: {predicted_label:.4f} (Actual Label: {label})")
            
        # Aggregate weights
        with torch.no_grad():
            for param, global_weight in zip(local_model.parameters(), global_weights):
                global_weight += param.data.clone() * num_samples
                
    # Update overall metrics
    TP_total += TP
    FP_total += FP
    TN_total += TN
    FN_total += FN
    
    # Calculate detection metrics for this round
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    false_detection_rate = FP / (FP + TP) if (FP + TP) > 0 else 0
    test_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Write metrics to CSV
    gat_metrics_writer.writerow([round_num + 1, TP, FP, TN, FN, FNR, FPR, detection_rate, false_detection_rate,
                                 test_accuracy, precision, recall, f1])
    
    # Calculate detection percentage
    total_attackers_appeared = len(actual_attackers)
    detection_percentage = (detected_attackers / total_attackers_appeared) * 100 if total_attackers_appeared > 0 else 0

    # Log detection summary for the round
    detection_summary_writer.writerow([round_num + 1, total_attackers_appeared, detected_attackers, detection_percentage])
    print(f"Round {round_num + 1} - Total Attackers Appeared: {total_attackers_appeared}, Detected Attackers: {detected_attackers}, Detection Percentage: {detection_percentage:.2f}%")

    # Print round summary
    print(f"Round {round_num + 1} - Average Training Accuracy: {np.mean(client_train_accuracies):.4f}, Average Training Loss: {np.mean(client_train_losses):.4f}")
     

    # Record potential and actual attackers in this round
    attackers_appearances_writer.writerow([round_num + 1, ', '.join(potential_attackers), ', '.join(actual_attackers)])

    # Calculate the total number of samples
    total_samples = np.sum(num_samples_per_client)
    print(f"\nTotal Number of Samples: {total_samples} ({' + '.join(map(str, num_samples_per_client))})")
    
    # Set the aggregated weights and biases for the global model
    with torch.no_grad():
        for global_weight in global_weights:
            global_weight /= total_samples  # Average the accumulated weights
    global_model.load_state_dict({name: global_weight for name, global_weight in zip(global_model.state_dict(), global_weights)})
    
    # Save global weights after aggregation
    after_aggregation_path = f"path/to/your/GAT_detection_folder/after_aggregation/{detection_threshold}_threshold_Attackers_{percentage_attackers}_alpha_{alpha}/Round_{round_num+1}.h5"
    os.makedirs(os.path.dirname(after_aggregation_path), exist_ok=True)
    with h5py.File(after_aggregation_path, 'w') as file:
        for layer_idx, layer in enumerate(global_model.children()):
            if hasattr(layer, 'weight') and layer.weight is not None:
                file.create_dataset(f'layer_{layer_idx}_weights', data=layer.weight.data.cpu().numpy())
            if hasattr(layer, 'bias') and layer.bias is not None:
                file.create_dataset(f'layer_{layer_idx}_biases', data=layer.bias.data.cpu().numpy())
                
    

    # Calculate and store the average training accuracy and average training loss for this round
    average_train_accuracies.append(np.mean(client_train_accuracies))
    average_train_losses.append(np.mean(client_train_losses))
    avg_train_acc_writer.writerow([round_num + 1, np.mean(client_train_accuracies), np.mean(client_train_losses)])
    
   
    # Evaluation on test data
    test_images = []
    test_labels = []

    # Iterate over class folders in the test data
    test_class_folders = os.listdir(test_data_path)
    for class_folder in test_class_folders:
        class_folder_path = os.path.join(test_data_path, class_folder)
        
        # Iterate over image files within the class folder
        image_files = os.listdir(class_folder_path)
        for image_file in image_files:
            image_path = os.path.join(class_folder_path, image_file)
            image = Image.open(image_path).convert('L')  # Convert image to grayscale
            transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])  # Resize and convert to tensor
            image_tensor = transform(image)
            test_images.append(image_tensor)
            test_labels.append(int(class_folder))

    # Convert test_images and test_labels to tensors
    x_test = torch.stack(test_images)
    y_test = torch.tensor(test_labels)

    # Perform predictions on the test data
    global_model.eval()
    with torch.no_grad():
        outputs = global_model(x_test)
        _, y_pred_labels = torch.max(outputs, 1)

    # Calculate test accuracy
    accuracy = (y_pred_labels == y_test).sum().item() / len(y_test)
    test_accuracies.append(accuracy)
    test_acc_writer.writerow([round_num + 1, accuracy])
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred_labels, average='macro', zero_division='warn')
    recall = recall_score(y_test, y_pred_labels, average='macro', zero_division='warn')
    f1 = f1_score(y_test, y_pred_labels, average='macro', zero_division='warn')

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print(f"\nRound {round_num + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    evaluation_metrics_writer.writerow([round_num + 1, precision, recall, f1])

# Calculate overall metrics
FNR_total = FN_total / (FN_total + TP_total) if (FN_total + TP_total) > 0 else 0
FPR_total = FP_total / (FP_total + TN_total) if (FP_total + TN_total) > 0 else 0
DR_total = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
FDR_total = FP_total / (FP_total + TP_total) if (FP_total + TP_total) > 0 else 0
overall_accuracy = (TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total) if (TP_total + TN_total + FP_total + FN_total) > 0 else 0
overall_precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
overall_recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

# Write overall metrics to CSV
gat_metrics_writer.writerow(["Overall", TP_total, FP_total, TN_total, FN_total, FNR_total, FPR_total, DR_total,
                             FDR_total, overall_accuracy, overall_precision, overall_recall, overall_f1])   

# Close the CSV files
csv_file.close()
avg_train_acc_file.close()
test_acc_file.close()
evaluation_metrics_file.close()
attackers_csv_file.close()
attackers_appearances_file.close()
gat_benign_prediction_file.close()
gat_attacker_prediction_file.close()
detection_summary_file.close()
gat_metrics_file.close()


# Print the total execution time
end_time = time.time()
total_time = end_time - start_time
hours = total_time // 3600
minutes = (total_time % 3600) // 60
seconds = total_time % 60
print(f"Total execution time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
