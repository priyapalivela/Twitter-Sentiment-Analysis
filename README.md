# Twitter Sentiment Analysis - NLP Pipeline

Binary sentiment classification (positive/negative) on Twitter data using 
a complete NLP pipeline with model serialization for production inference.

##  Results
- Logistic Regression with GridSearchCV hyperparameter tuning
- Evaluated with accuracy, precision, confusion matrix
- Model serialized via pickle for inference on new tweets

##  Tech Stack
- **NLP:** CountVectorizer, text preprocessing, stopword removal
- **Model:** Logistic Regression (L2 regularization, GridSearchCV)
- **Evaluation:** Accuracy, Precision, Confusion Matrix
- **Libraries:** Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Pickle

##  Project Structure
```
├── Twitter_Sentiment_Analysis.ipynb  # Full pipeline
├── vectorizer.pkl                     # Saved CountVectorizer
├── model.pkl                          # Saved Logistic Regression model
├── data/                              # Dataset
└── images/                            # Visualizations
```

##  Pipeline
1. Text preprocessing — lowercasing, punctuation removal, stopwords
2. CountVectorizer feature extraction
3. Logistic Regression with GridSearchCV (C, penalty, solver)
4. Feature importance analysis via model coefficients
5. Model + vectorizer saved with pickle

## 👩‍💻 Author
Bhanu Priya Palivela | [GitHub](https://github.com/priyapalivela)
