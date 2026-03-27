# Comment-Toxicity-
A professional, deep learning-powered solution for identifying toxic content in online comments. This project leverages state-of-the-art Natural Language Processing (NLP) techniques and Deep Learning architectures to provide real-time toxicity analysis and category-wise risk assessment.
🚀 Overview
In the age of digital interaction, maintaining a healthy online environment is crucial. This project provides a robust dashboard that classifies user comments into six different categories of toxicity:

Toxic
Severe Toxic
Obscene
Threat
Insult
Identity Hate
The core of this system is a Bidirectional LSTM (Long Short-Term Memory) network, which outperformed traditional CNN models during our rigorous benchmarking phase.

📊 Dataset Insight
The models were trained on a large-scale dataset of Wikipedia comments, specifically processed for multi-label classification.

Training Samples: 159,571 comments
Testing Samples: 153,164 comments
Class Distribution: Highly imbalanced data with a significant majority of "Clean" comments (~90%), requiring specialized training strategies.
🏗️ Technical Architecture
1. Preprocessing Pipeline
Our pipeline ensures that every comment is cleaned and transformed into a format optimal for deep learning:

Text Cleaning: Removal of URLs, special characters, and noise using regular expressions.
Tokenization: Mapping words to a vocabulary of 20,000 unique integers.
Padding: Standardizing sequence lengths to 150 tokens for consistent model input.
OOV Handling: Intelligent handling of Out-Of-Vocabulary words.
2. Model Benchmarking (CNN vs. LSTM)
We implemented and compared two high-performance architectures:

Metric	CNN Model	LSTM Model (Winner)
Mean Accuracy	0.9712	0.9739
Contextual Awareness	Localized (n-grams)	Global (Sequential dependencies)
Complexity	Lower	Higher
The Best Choice: The Bidirectional LSTM was selected because it processes text in both directions (forward and backward), allowing it to capture the subtle context and dependencies between words that simple convolutional filters might miss.

3. Training Strategy
Optimizer: Adam (Adaptive Moment Estimation) for efficient learning.
Loss Function: Binary Crossentropy (ideal for multi-label tasks).
Regulation: SpatialDropout1D and EarlyStopping to prevent overfitting and ensure generalization.
🛠️ Tech Stack
Framework: Streamlit (for the interactive dashboard)
Deep Learning: TensorFlow & Keras
Data Science: Pandas, NumPy, Scikit-learn
Visualization: Matplotlib, Seaborn
deployment: Python 3.x
🏁 Getting Started
Prerequisites
Python 3.8+
Required libraries listed in requirements.txt (or install manually: streamlit, tensorflow, pandas, numpy, matplotlib, seaborn)
Execution
Clone the repository.
Ensure best_toxicity_model.h5 (or lstm_model.keras) and tokenizer.pkl are in the root directory.
Run the application:
streamlit run app.py
🎯 Project Impact
This dashboard translates complex neural network probabilities into actionable safety scores, empowering moderators and community managers to make fast, data-driven decisions to protect online spaces.
