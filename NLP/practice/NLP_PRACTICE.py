import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

#Load train and test data into their respective data frames

train_df = pd.read_csv('/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/train.csv')
test_df = pd.read_csv('/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/test.csv')

#now that our data is loaded into their respective data frames, lets take a look at what they contain 

value = train_df[train_df["target"] ==0]["text"].values[1]
print(value)

# and now a negative text

value2 = train_df[train_df["target"] == 1]["text"].values[1]
print(value2)

# Building vetors 

    # The theory behind the model we'll build in this notebook is pretty simple: the words contained in each tweet are a good indicator of whether they're about a real disaster or not (this is not entirely correct,
    #  but it's a great place to start).
    # We'll use scikit-learn's CountVectorizer to count the words in each tweet and turn them into data our machine learning model can process.
    # Note: a vector is, in this context, a set of numbers that a machine learning model can work with. We'll look at one in just a second.

count_vectorizer = feature_extraction.text.CountVectorizer() # Convert a collection of text documents to a matrix of token counts.

# Lets get counts fro the first 5 tweets in the data 
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])

# we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)
print(f"{example_train_vectors[0].todense().shape} \n")
print(f"{example_train_vectors[0].todense()} \n")


# The above tells us that:

# There are 54 unique words (or "tokens") in the first five tweets.
# The first tweet contains only some of those unique tokens - all of the non-zero counts above are the tokens that DO exist in the first tweet.
# Now let's create vectors for all of our tweets.

train_vectors = count_vectorizer.fit_transform(train_df["text"])

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 
# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(test_df["text"])

# building model

## Our vectors are really big, so we want to push our model's weights
## toward 0 without completely discounting different words - ridge regression 
## is a good way to do this.

clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
print(f"\n {scores} \n")

clf.fit(train_vectors, train_df["target"])

sample_submission = pd.read_csv("/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)
print(sample_submission.head())