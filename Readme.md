
# Emotion Classifier and Alert System

This project implements an emotion classification system using Support Vector Machines (SVM) and TF-IDF vectorization. It includes a simple alert system for detecting high-risk emotions.

## Project Structure

The main script performs the following steps:
1. Loads and preprocesses the dataset.
2. Extracts features using TF-IDF.
3. Trains a baseline linear SVM model.
4. Performs a Grid Search to find the best SVM kernel and hyperparameters.
5. Evaluates the best performing model.
6. Demonstrates a basic emotion alert system.
7. Saves the trained model and related components.

## Setup and Installation

1. **Environment:** This code is designed to run in a Python environment, preferably within a Google Colab notebook or a Jupyter Notebook.
2. **Libraries:** The project uses standard data science libraries. Install them using pip:

## Usage

1. **Data Preparation:** Ensure your data is in a CSV file with at least two columns: 'text' containing the input text and 'label' containing the emotion label for each text.
2. **Upload Data:** If you are using Google Colab, upload your CSV data file to the Colab environment (e.g., in the `/content/` directory).
3. **Update Data Path:** In the code, locate the `main` function and update the `data_path` variable to point to your uploaded CSV file:

4. **Run the Notebook:** Execute all the code cells in the notebook sequentially.
5. **Output:** The script will print progress, model evaluation results, and a demonstration of the alert system. The trained model, TF-IDF vectorizer, and label encoder will be saved to the specified `output_file` (default is `emotion_classifier.pkl`).

## Code Explanation

- `load_data(path)`: Loads the CSV data and provides initial statistics.
- `preprocess_and_encode(df)`: Handles missing values and converts categorical labels into numerical representations using `LabelEncoder`.
- `extract_features(df, max_features=5000)`: Transforms the text data into numerical features using `TfidfVectorizer`.
- `evaluate_model(model, X_test, y_test, label_encoder)`: Calculates and prints evaluation metrics (accuracy, classification report) and displays a confusion matrix.
- `EmotionAlertSystem`: A class that takes the trained model and vectorizer to predict emotion and confidence for new text, and triggers alerts based on defined rules and a confidence threshold.
- `main(args)`: Orchestrates the entire process from data loading to model saving.

## Alert System

The `EmotionAlertSystem` uses predefined rules to categorize emotions as 'high_risk' or 'medium_risk'. If the model predicts one of these emotions with a confidence exceeding a specified `threshold`, an alert is generated. You can modify the `rules` dictionary in the `main` function to customize the risk categories and threshold.

## Saved Model

The trained pipeline (best performing SVM model, TF-IDF vectorizer, label encoder, and alert rules) is saved as a pickle file (`emotion_classifier.pkl`) using `joblib`. This file can be loaded later to make predictions on new data without retraining the model.

## Future Improvements

- Explore other classification algorithms (e.g., Naive Bayes, Logistic Regression, deep learning models).
- Experiment with different text preprocessing techniques (e.g., stemming, lemmatization, removing punctuation).
- Optimize TF-IDF parameters or explore other feature extraction methods (e.g., word embeddings).
- Implement a more sophisticated alert system with customizable actions or notifications.
- Evaluate the model on a larger and more diverse dataset.
