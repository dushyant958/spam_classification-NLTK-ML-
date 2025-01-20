# Spam-Ham Classification Project

## Project Overview
This project is a Spam-Ham classifier implemented using the Bag of Words and TF-IDF methods for feature extraction. The classification is performed using the Naive Bayes Machine Learning method, specifically the Multinomial Naive Bayes class. The project leverages the Natural Language Toolkit (NLTK) library for text preprocessing and model implementation.

## Features
- **Data Preprocessing**: Tokenization, stopword removal, and text normalization using NLTK.
- **Feature Extraction**:
  - Bag of Words (BoW)
  - Term Frequency-Inverse Document Frequency (TF-IDF)
- **Model Training**: Using Multinomial Naive Bayes for classification.
- **Evaluation**: Model evaluation using accuracy, precision, recall, and F1-score.

## Dependencies
- Python 3.x
- NLTK
- Scikit-learn
- Pandas
- Jupyter Notebook

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-ham-classifier.git
   ```
2. Navigate to the project directory:
   ```bash
   cd spam-ham-classifier
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Load the `spam_ham_classifier.ipynb` notebook.
3. Run the cells sequentially to preprocess data, extract features, train the model, and evaluate its performance.

## Project Structure
- `data/`: Contains the dataset used for classification.
- `notebooks/`: Jupyter notebooks with the project code.
- `models/`: Directory for saving trained models.
- `requirements.txt`: List of dependencies.

## Key Functions and Methods
- **Text Preprocessing**: Utilizes NLTK for tokenization, removing stopwords, and stemming/lemmatization.
- **Feature Extraction**: Implements both Bag of Words and TF-IDF vectorization.
- **Model Training**: Uses Scikit-learn's `MultinomialNB` for Naive Bayes classification.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score calculated using Scikit-learn.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Jupyter Notebook](https://jupyter.org/)

