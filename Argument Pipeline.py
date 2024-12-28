# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:01:11 2024

@author: coryf
"""

import pandas as pd
import string
import nltk
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import statistics
import scipy
from scipy.stats import skew
from lexical_diversity import lex_div
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score

#%%
# Load Data
df = pd.read_csv('deceptive-opinion.csv')

# Create truth value variable
df['truth_value'] = df['deceptive'].apply(lambda x: 0 if x == 'deceptive' else 1)

#%%
# Remove punctuation


def remove_punc(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


df['text_no_punc'] = df['text'].apply(remove_punc)

#%%
# Make letters lowercase


def make_lowercase(text):
    text = text.lower()
    return text


df['text_lowercase'] = df['text_no_punc'].apply(make_lowercase)

#%%
# Tokenize statements


def tokenize(text):
    text = nltk.word_tokenize(text)
    return text


df['text_tokens'] = df['text_no_punc'].apply(nltk.word_tokenize)

#%%
# tag parts of speech of text

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
df['text_pos_tags'] = df['text_tokens'].apply(nltk.pos_tag)

#%%
# Lemmatize words

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_words(text):
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) 
                        for w in words]
    lemmatized_text = " ".join(lemmatized_words)
    return lemmatized_text


df['lemmatized_words'] = df['text_lowercase'].apply(lemmatize_words)

#%%
# Print examples

print(df.loc[18].at['text'])
print(df.loc[18].at['text_no_punc'])
print(df.loc[18].at['text_lowercase'])
print(df.loc[18].at['text_tokens'])
print(df.loc[18].at['text_pos_tags'])
print(df.loc[18].at['lemmatized_words'])

df.info()

df.to_csv('Processed Reviews.csv', index=False)

#%%
# Create word count variable

df['word_count'] = df['text_no_punc'].apply(lambda x: len(x.split()))

# Calculate statistics for number of word count

min_words = min(df['word_count'])
median_words = statistics.median(df['word_count'])
mean_words = df['word_count'].mean()
std_words = statistics.pstdev(df['word_count'])
max_words = max(df['word_count'])

print("Word Count Statistics")
print("Minimum:", min_words)
print("Median:", median_words)
print("Max:", max_words)
print("Mean:", mean_words)
print("Standard Deviation:", std_words)
print("Skewness:", skew(df['word_count'], axis=0, bias=True))

#%%
# Create exclusive word variable


def num_excl_words(text):
    words = text.lower().split()
    exclusive_words = [
        "but", "except", "without", "however", "though", "unless",
        "while", "although", "whereas", "yet", "aside", "besides",
        "excluding", "apart", "other than", "instead"
    ]
    matching_count = sum(1 for word in words if word in exclusive_words)
    return matching_count


df['excl_word_count'] = df['text_lowercase'].apply(num_excl_words)

# Calculate statistics for exclusive words

min_words = min(df['excl_word_count'])
median_words = statistics.median(df['excl_word_count'])
mean_words = df['excl_word_count'].mean()
std_words = statistics.pstdev(df['excl_word_count'])
max_words = max(df['excl_word_count'])

print("Exlcusive Word Count Statistics")
print("Minimum:", min_words)
print("Median:", median_words)
print("Max:", max_words)
print("Mean:", mean_words)
print("Standard Deviation:", std_words)
print("Skewness:", skew(df['excl_word_count'], axis=0, bias=True))

#%%
# Create personal pronoun variable


def num_prps(text):
    prps_count = sum(1 for word, tag in text if tag in ['PRP', 'PRP$'])
    return prps_count


df['prps_tag_count'] = df['text_pos_tags'].apply(num_prps)

# Calculate statistics for number of personal pronouns

min_words = min(df['prps_tag_count'])
median_words = statistics.median(df['prps_tag_count'])
mean_words = df['prps_tag_count'].mean()
std_words = statistics.pstdev(df['prps_tag_count'])
max_words = max(df['prps_tag_count'])

print("Personal Pronoun Count Statistics")
print("Minimum:", min_words)
print("Median:", median_words)
print("Max:", max_words)
print("Mean:", mean_words)
print("Standard Deviation:", std_words)
print("Skewness:", skew(df['prps_tag_count'], axis=0, bias=True))

#%%
# Load NRC Emotion Lexicon

lexicon = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
                      sep='\t', header=None,
                      names=['Word', 'Emotion', 'Association'])

negative_words = lexicon[(lexicon['Emotion'] == 'negative') &
                         (lexicon['Association'] == 1)]['Word'].tolist()

anger_words = lexicon[(lexicon['Emotion'] == 'anger') &
                      (lexicon['Association'] == 1)]['Word'].tolist()


def combine_lists(list1, list2):
    combined_list = list1 + list2
    combined_list.sort()
    return combined_list


combined_list = combine_lists(negative_words, anger_words)

#%%
# Create negative word variable


def num_neg_words(text):
    words = text.lower().split()
    matching_count = sum(1 for word in words if word in combined_list)
    return matching_count


df['neg_word_count'] = df['lemmatized_words'].apply(num_neg_words)

# Calculate statistics of number of negative words

min_words = min(df['neg_word_count'])
median_words = statistics.median(df['neg_word_count'])
mean_words = df['neg_word_count'].mean()
std_words = statistics.pstdev(df['neg_word_count'])
max_words = max(df['neg_word_count'])

print("Negative Word Count Statistics")
print("Minimum:", min_words)
print("Median:", median_words)
print("Max:", max_words)
print("Mean:", mean_words)
print("Standard Deviation:", std_words)
print("Skewness:", skew(df['neg_word_count'], axis=0, bias=True))

#%%
# Create lexical diversity variable


df['lexical_diversity'] = df['text'].apply(lex_div.mtld)

# Calculate statistics of lexical diversity

min_words = min(df['lexical_diversity'])
median_words = statistics.median(df['lexical_diversity'])
mean_words = df['lexical_diversity'].mean()
std_words = statistics.pstdev(df['lexical_diversity'])
max_words = max(df['lexical_diversity'])

print("Lexical Diversity statistics")
print("Minimum:", min_words)
print("Median:", median_words)
print("Max:", max_words)
print("Mean:", mean_words)
print("Standard Deviation:", std_words)
print("Skewness:", skew(df['lexical_diversity'], axis=0, bias=True))

#%%

df.info()

df.to_csv('Processed Reviews with Features.csv', index=False)

#%%
# Create histograms

plt.hist(df['word_count'], bins=20, color='blue', edgecolor='black', alpha=0.7)

plt.title('Histogram of Word Counts')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.show()

plt.hist(df['excl_word_count'],bins=13, color='blue',
         edgecolor='black', alpha=0.7)

plt.title('Histogram of Exclusive Word Counts')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.show()

plt.hist(df['prps_tag_count'], bins=20, color='blue',
         edgecolor='black', alpha=0.7)

plt.title('Histogram of Personal Pronoun Counts')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.show()

plt.hist(df['neg_word_count'], bins=36, color='blue',
         edgecolor='black', alpha=0.7)
plt.title('Histogram of Negative Word Counts')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.show()

plt.hist(df['lexical_diversity'], bins=20, color='blue',
         edgecolor='black', alpha=0.7)

plt.title('Histogram of Lexical Diversity')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.show()

 #%%
# Create scatterplots

plt.scatter(df['deceptive'], df['word_count'], color='blue', marker='o')
plt.title("Word Count vs. Truth-Value")
plt.show()

plt.scatter(df['deceptive'], df['excl_word_count'], color='blue', marker='o')
plt.title("Exclusive Word Count vs. Truth-Value")
plt.show()

plt.scatter(df['deceptive'], df['prps_tag_count'], color='blue', marker='o')
plt.title("Personal Pronoun Count vs. Truth-Value")
plt.show()

plt.scatter(df['deceptive'], df['neg_word_count'], color='blue', marker='o')
plt.title("Negative Word Count vs. Truth-Value")
plt.show()

plt.scatter(df['deceptive'], df['lexical_diversity'], color='blue', marker='o')
plt.title("Lexical Diversity vs. Truth-Value")
plt.show()



#%%
# Randomly split data into testing and training data

testing = df.sample(frac = 0.2)
training = df.drop(testing.index)

training.info()
testing.info()

df.to_csv('Testing Reviews.csv', index=False)
df.to_csv('Training Reviews.csv', index=False)

#%%
# Calculate mutual information of features

mi_word_count = mutual_info_score(training['truth_value'], training['word_count'])
print("Mutual Information of Word Count and Truth-value:", mi_word_count)

mi_excl_word_count = mutual_info_score(training['truth_value'],
                                       training['excl_word_count'])
print("Mutual Information of Exclusive Word Count and Truth-value:",
      mi_excl_word_count)

mi_prps_tag_count = mutual_info_score(training['truth_value'], training['prps_tag_count'])
print("Mutual Information of Personal Pronoun Count and Truth-value:",
      mi_prps_tag_count)

mi_neg_word_count = mutual_info_score(training['truth_value'], training['neg_word_count'])
print("Mutual Information of Neg Word Count and Truth-value:", 
      mi_neg_word_count)

# Perform likelihood cross-validation to determine optimum number of neighbors for continuous probabilities


def likelihood_cross_validation(discrete_var, continuous_var, n_neighbors_values, cv_folds=5):

    discrete_var = np.array(discrete_var).reshape(-1, 1)
    continuous_var = np.array(continuous_var)
    n_samples = len(discrete_var)

    # Calculate the maximum possible value of n_neighbors for the smallest fold
    min_fold_size = n_samples * (cv_folds - 1) // cv_folds
    valid_neighbors = [n for n in n_neighbors_values if n < min_fold_size]
    if not valid_neighbors:
        raise ValueError("No valid n_neighbors values. Reduce cv_folds or add smaller n_neighbors values.")

    # Set up cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    mi_scores = []

    for n_neighbors in valid_neighbors:
        fold_scores = []
        for train_idx, test_idx in kf.split(discrete_var):
            discrete_train, continuous_train = discrete_var[train_idx], continuous_var[train_idx]

            # Train mutual information on train set
            mi = mutual_info_regression(
                discrete_train, continuous_train,
                n_neighbors=n_neighbors, random_state=42
            )
            fold_scores.append(mi[0])  # Only one feature, so use the first value.

        # Average score for this value of n_neighbors
        mi_scores.append(np.mean(fold_scores))

    # Find the best number of neighbors
    optimal_n_neighbors = valid_neighbors[np.argmax(mi_scores)]
    return optimal_n_neighbors, mi_scores


n_neighbors_values = [2, 3, 4, 5, 6, 7]

optimal_n_neighbors, mi_scores = likelihood_cross_validation(training['truth_value'], training['lexical_diversity'], n_neighbors_values)
print(f"Optimal n_neighbors: {optimal_n_neighbors}")

# Calculate mutual information of lexical diversity


def calculate_mutual_information(discrete_var, continuous_var, n_neighbors=4):
    discrete_var = np.array(discrete_var).reshape(-1, 1)
    continuous_var = np.array(continuous_var)

    # Estimate mutual information
    mi = mutual_info_regression(discrete_var, continuous_var, n_neighbors=n_neighbors, random_state=0)
    return mi[0]


mi_lex_div = calculate_mutual_information(training['truth_value'],
                                          training['lexical_diversity'])
print(f"Mutual Information of Lexical Diversity and Truth-value: {mi_lex_div}")

#%%
# Create vector of mutual informations

mut_info = np.array([mi_word_count, mi_excl_word_count, mi_prps_tag_count, mi_neg_word_count, mi_lex_div])

#%%
# Calculate association
# This may not work for continuous variables

def r_correlation(vector):
    mut_info_x2 = mut_info * 2
    mut_info_xminus2 = mut_info_x2 * (-1)
    mut_info_e = np.exp(mut_info_xminus2)
    one_minus_mut_info = 1 - mut_info_e
    sqrt_mut_info = np.sqrt(one_minus_mut_info)
    return sqrt_mut_info


measure = r_correlation(mut_info)
measure = measure / sum(measure)
print(measure)

#%%
# Calculate probability of deception for features

X_train = training[['word_count']]
X_test = testing[['word_count']]
y_train = training['truth_value']
y_test = testing['truth_value']

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

word_count_probs = model.predict_proba(X_test)

word_count_deception_probs = word_count_probs[:, 0]


X_train = training[['excl_word_count']]
X_test = testing[['excl_word_count']]

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

excl_word_count_probs = model.predict_proba(X_test)

excl_word_count_deception_probs = excl_word_count_probs[:, 0]


X_train = training[['prps_tag_count']]
X_test = testing[['prps_tag_count']]

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

prps_tag_count_probs = model.predict_proba(X_test)

prps_tag_count_deception_probs = prps_tag_count_probs[:, 0]


X_train = training[['neg_word_count']]
X_test = testing[['neg_word_count']]

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

neg_word_count_probs = model.predict_proba(X_test)

neg_word_count_deception_probs = neg_word_count_probs[:, 0]


X_train = training[['lexical_diversity']]
X_test = testing[['lexical_diversity']]

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

lex_div_probs = model.predict_proba(X_test)

lex_div_deception_probs = lex_div_probs[:, 0]

#%%
# Combine probability vectors into dataframe

probs = pd.DataFrame({"word_count": word_count_deception_probs, "excl_word_count": excl_word_count_deception_probs, "prps_tag_count": prps_tag_count_deception_probs, "neg_word_count": neg_word_count_deception_probs, "lexical_diversity": lex_div_deception_probs})

# calculate final probabilities

final_probs = np.dot(probs, measure)
testing['final_probs'] = final_probs
testing['guess'] = (testing['final_probs'] < 0.5).astype(int)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(testing['truth_value'], testing['guess'])
precision = precision_score(testing['truth_value'], testing['guess'])
recall = recall_score(testing['truth_value'], testing['guess'])

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


























#%%
# Calculate probabilities of features


def calculate_probabilities(data):
    total_count = len(data)
    if total_count == 0:
        raise ValueError("Data must contain at least one value.")

    value_counts = Counter(data)
    probabilities = [count / total_count for count in value_counts.values()]
    return probabilities


word_count_probs = calculate_probabilities(df['word_count'])
excl_word_count_probs = calculate_probabilities(df['excl_word_count'])
prps_tag_count_probs = calculate_probabilities(df['prps_tag_count'])
neg_word_count_probs = calculate_probabilities(df['neg_word_count'])

probs = calculate_probabilities(df['deceptive'])

#%%
# Calculate entropy of features

word_count_entropy = entropy(word_count_probs, base=2)
print("Word Count Entropy:", word_count_entropy)
num_excl_words_entropy = entropy(excl_word_count_probs, base=2)
print("Exclusive Word Count Entropy:", num_excl_words_entropy)
prps_tag_entropy = entropy(prps_tag_count_probs, base=2)
print("Personal Pronoun Count Entropy:", prps_tag_entropy)
neg_word_entropy = entropy(neg_word_count_probs, base=2)
print("Negative Word Count Entropy:", neg_word_entropy)
truth_entropy = entropy(probs, base=2)
print("Truth-Value Entropy:", truth_entropy)


def continuous_entropy(data):
    hist, bin_edges = np.histogram(data, bins='auto', density=True)
    return entropy(hist)


lexical_entropy = continuous_entropy(df['lexical_diversity'])
print("Lexical Diversity Entropy:", lexical_entropy)

