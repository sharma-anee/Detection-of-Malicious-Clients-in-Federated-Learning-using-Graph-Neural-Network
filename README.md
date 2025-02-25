# Overview
This repository contains the implementation of malicious client detection in Federated Learning (FL) using Graph Neural Networks (GNNs). The approach utilizes Graph Attention Networks (GAT) to classify FL clients as benign or attackers based on graph representations of their local model updates.

## **Main Features**
- Federated Learning Simulation: Clients train local models and send updates for aggregation.
- Graph Representation of FL Models: Converts CNN weights and biases into structured graphs.
- GAT-Based Anomaly Detection: A pre-trained GAT model classifies clients based on their graph structures.
- Sign Flipping Attack Simulation: Attackers flip model parameters with a defined probability.
- Performance Logging: Tracks detection accuracy, precision, recall, F1-score, and false detection rate.
## **How to Use**
- Modify Paths: Update dataset and model paths in the script.
- Set Experiment Parameters: Input the number of FL rounds, clients, attacker percentage, attack probability and threshold for detection.
- Run the Script: Execute the script to start FL training and anomaly detection.
## **Outputs**
The script generates CSV logs containing:

- Evaluation metrics (average train accuracy, loss, test accuracy etc) of the FL local models and the FL global model
- Detection metrics (TP, FP, FN, TN, detection rate, false detection rate, etc.) of the GAT detection model.
- GAT model predictions for benign and attacker clients

## **Future Improvements**
- Evaluation on more real-world datasets
- Adaptive attack strategies for dynamic adversarial behavior
- Latency and resource profiling for deployment in edge environments
#### For more details, refer to the code comments and logs generated during execution.
