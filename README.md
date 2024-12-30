# Chatbot for Yes-No Questions

## Overview
This project implements a chatbot capable of answering yes-no questions using an End-to-End Memory Network (MemN2N) model. The MemN2N model, introduced in the paper [\"End-To-End Memory Networks\"](https://arxiv.org/abs/1503.08895) by Sainbayar Sukhbaatar et al., is a neural network architecture designed to handle tasks requiring memory and reasoning. It encodes input sequences and questions into embeddings, computes relationships between them using attention mechanisms, and predicts answers based on learned representations.

The chatbot is trained using a synthetic QA dataset derived from the [bAbI dataset](https://www.kaggle.com/datasets/roblexnana/the-babi-tasks-for-nlp-qa-system), which provides structured QA tasks for evaluating reasoning models. The chatbot is specifically optimized for yes-no question answering.

---

## Key Features
- **End-to-End Memory Network Implementation**: Utilizes MemN2N to perform memory-based reasoning for QA tasks.
- **Attention Mechanism**: Employs attention to compute relevance between story context and questions.
- **Synthetic Dataset**: Trains on the bAbI dataset, specifically tailored for yes-no question answering tasks.
- **Customizable Architecture**: The model can be fine-tuned or adapted for other QA tasks beyond yes-no questions.
- **Interactive Interface**: Provides a foundation for building a user-friendly chatbot capable of responding to natural language inputs.

---

## Key Steps
1. **Dataset Preparation**:
   - Download the bAbI dataset from [Kaggle](https://www.kaggle.com/datasets/roblexnana/the-babi-tasks-for-nlp-qa-system).
   - Preprocess the dataset to focus on yes-no questions.

2. **Model Development**:
   - Define the input encoders for story context and questions.
   - Implement the attention mechanism using dot product and softmax activation.
   - Combine encoded representations and pass them through LSTM layers to predict answers.

3. **Training**:
   - Train the MemN2N model using the preprocessed dataset.
   - Evaluate the model's performance on validation and test sets.

4. **Deployment**:
   - Save the trained model using Keras or TensorFlow utilities.
   - Build an interactive interface for the chatbot to handle real-time user inputs.

---

## Project Limitations
- **Synthetic Dataset**: The model is trained on synthetic data (bAbI dataset), which may not generalize well to real-world natural language inputs.
- **Limited Scope**: Focuses only on yes-no questions, making it less versatile for broader conversational AI applications.
- **Performance on Complex Contexts**: The MemN2N model may struggle with complex, ambiguous, or out-of-scope questions.
- **Resource Intensive**: Training the model requires significant computational resources, especially for larger datasets or higher-dimensional embeddings.

---

