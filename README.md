# Text Feature Engineering Summary
## Overview
This lab implements text feature extraction for the Goodreads reviews dataset stored in the Gold layer. The goal is to transform raw review text into structured numerical features for downstream machine-learning models.

## 1. Dataset Preparation
The curated Gold dataset was loaded from feature_v2/train. All processing was applied to the cleaned review text column. The train, validation, and test splits were created earlier to prevent data leakage.

## 2. Text Cleaning
The review text was normalized before feature extraction:
1. converted to lowercase
2. punctuation removed
3. URLs and numbers replaced
4. extra whitespace trimmed
5. very short reviews removed
This ensured consistent input for all feature engineering steps.

## 3. Core Feature Extraction
### 3.1 Review Length Features
Two metrics that capture writing style:
1. review_length_words
2. review_length_chars

### 3.2 Sentiment Features
Using NLTK’s VADER sentiment analyzer:
1. sentiment_pos
2. sentiment_neg
3. sentiment_neu
4. sentiment_compound
These capture the emotional tone of the text.

### 3.3 TF-IDF Features

TF-IDF vectors were created using scikit-learn:
1. fitted only on the training split
2. unigrams and bigrams
3. stop-word removal
4. vocabulary size capped
This produces a sparse representation of word importance.

### 3.4 Semantic Embeddings (Sentence-BERT)
Semantic embeddings were generated using the model all-MiniLM-L6-v2.
Each review was transformed into a dense 384-dimensional vector that captures contextual meaning.

## 4. Additional Custom Features
### 4.1 Word Diversity
Measures vocabulary variation:
`unique_words / total_words`

### 4.2 Emotion Punctuation Score
Captures emotional expression through punctuation:
`(exclamation_count + question_mark_count) / (text_length + 1)`

### 4.3 Repeated Word Ratio
Measures redundancy in writing:
`(total_words - unique_words) / (total_words + 1)`

## 5. Final Feature Matrix
All engineered features were combined into one final dataset:
1. length features
2. sentiment features
3. TF-IDF vectors
4. semantic embeddings
5. additional custom features
6. metadata: `review_id`, `book_id`, `rating`

The combined dataset was saved in the Gold layer under features_v2/final_features and will be used for model training in the next lab.
