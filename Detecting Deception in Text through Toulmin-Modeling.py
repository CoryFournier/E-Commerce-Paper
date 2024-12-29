# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:01:11 2024

@author: coryf
"""

import pandas as pd
import string
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import statistics
import scipy.stats as stats
from scipy.stats import skew, entropy, pearsonr, spearmanr, ttest_ind
from lexical_diversity import lex_div
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import mutual_info_score, classification_report
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
# Create wordcloud of column of texts

all_text = " ".join(df['lemmatized_words'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()

#%%
# Create word count variable and calculate statistics

df['word_count'] = df['text_no_punc'].apply(lambda x: len(x.split()))

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
# Create exclusive word variable and calculate statistics


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
# Create personal pronoun variable and calculate statistics


def num_prps(text):
    prps_count = sum(1 for word, tag in text if tag in ['PRP', 'PRP$'])
    return prps_count


df['pers_pro_count'] = df['text_pos_tags'].apply(num_prps)

min_words = min(df['pers_pro_count'])
median_words = statistics.median(df['pers_pro_count'])
mean_words = df['pers_pro_count'].mean()
std_words = statistics.pstdev(df['pers_pro_count'])
max_words = max(df['pers_pro_count'])

print("Personal Pronoun Count Statistics")
print("Minimum:", min_words)
print("Median:", median_words)
print("Max:", max_words)
print("Mean:", mean_words)
print("Standard Deviation:", std_words)
print("Skewness:", skew(df['pers_pro_count'], axis=0, bias=True))

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
# Create negative word variable and calculate statistics


def num_neg_words(text):
    words = text.lower().split()
    matching_count = sum(1 for word in words if word in combined_list)
    return matching_count


df['neg_word_count'] = df['lemmatized_words'].apply(num_neg_words)

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
# Create lexical diversity variable and calculate statistics


df['lexical_diversity'] = df['text'].apply(lex_div.mtld)

min_words = min(df['lexical_diversity'])
median_words = statistics.median(df['lexical_diversity'])
mean_words = df['lexical_diversity'].mean()
std_words = statistics.pstdev(df['lexical_diversity'])
max_words = max(df['lexical_diversity'])

print("Lexical Diversity Statistics")
print("Minimum:", min_words)
print("Median:", median_words)
print("Max:", max_words)
print("Mean:", mean_words)
print("Standard Deviation:", std_words)
print("Skewness:", skew(df['lexical_diversity'], axis=0, bias=True))

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

plt.hist(df['pers_pro_count'], bins=20, color='blue',
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
# Perform lognormal test

statistic, p_value = stats.shapiro(np.log(df['word_count']))
print(f"Shapiro-Wilk Test: Statistic = {statistic}, p-value = {p_value}")

statistic, p_value = stats.shapiro(np.log(df['excl_word_count']+1))
print(f"Shapiro-Wilk Test: Statistic = {statistic}, p-value = {p_value}")

statistic, p_value = stats.shapiro(np.log(df['pers_pro_count']+1))
print(f"Shapiro-Wilk Test: Statistic = {statistic}, p-value = {p_value}")

statistic, p_value = stats.shapiro(np.log(df['neg_word_count']+1))
print(f"Shapiro-Wilk Test: Statistic = {statistic}, p-value = {p_value}")

df['log_word_count'] = np.log(df['word_count'])

#%%
# Create Spearman correlation matrix

variable_df = df[['log_word_count', 'excl_word_count', 'pers_pro_count', 'neg_word_count', 'lexical_diversity']].copy()

spear_corr_matrix = variable_df.corr(method='spearman')

print("Spearman's Rank Correlation Matrix:")
print(spear_corr_matrix)

print(spear_corr_matrix['log_word_count'])
print(spear_corr_matrix['excl_word_count'])
print(spear_corr_matrix['pers_pro_count'])
print(spear_corr_matrix['neg_word_count'])
print(spear_corr_matrix['lexical_diversity'])

#%%
# Create scatterplots

plt.scatter(df['deceptive'], df['log_word_count'], color='blue', marker='o')
plt.title("Word Count vs. Truth-Value")
plt.show()

plt.scatter(df['deceptive'], df['excl_word_count'], color='blue', marker='o')
plt.title("Exclusive Word Count vs. Truth-Value")
plt.show()

plt.scatter(df['deceptive'], df['pers_pro_count'], color='blue', marker='o')
plt.title("Personal Pronoun Count vs. Truth-Value")
plt.show()

plt.scatter(df['deceptive'], df['neg_word_count'], color='blue', marker='o')
plt.title("Negative Word Count vs. Truth-Value")
plt.show()

plt.scatter(df['deceptive'], df['lexical_diversity'], color='blue', marker='o')
plt.title("Lexical Diversity vs. Truth-Value")
plt.show()

#%%
# Perform two-sample t-tests

group1 = df[df['truth_value'] == 1]['log_word_count']
group2 = df[df['truth_value'] == 0]['log_word_count']

t_stat, p_value = ttest_ind(group1, group2)

print("t-statistic:", t_stat)
print("p-value:", p_value)

group1 = df[df['truth_value'] == 1]['excl_word_count']
group2 = df[df['truth_value'] == 0]['excl_word_count']

t_stat, p_value = ttest_ind(group1, group2)

print("t-statistic:", t_stat)
print("p-value:", p_value)

group1 = df[df['truth_value'] == 1]['pers_pro_count']
group2 = df[df['truth_value'] == 0]['pers_pro_count']

t_stat, p_value = ttest_ind(group1, group2)

print("t-statistic:", t_stat)
print("p-value:", p_value)

group1 = df[df['truth_value'] == 1]['neg_word_count']
group2 = df[df['truth_value'] == 0]['neg_word_count']

t_stat, p_value = ttest_ind(group1, group2)

print("t-statistic:", t_stat)
print("p-value:", p_value)

group1 = df[df['truth_value'] == 1]['lexical_diversity']
group2 = df[df['truth_value'] == 0]['lexical_diversity']

t_stat, p_value = ttest_ind(group1, group2)

print("t-statistic:", t_stat)
print("p-value:", p_value)

#%%
# Calculate biserial correlation

def biserial_correlation(binary, continuous):
    # Ensure inputs are numpy arrays
    binary = np.array(binary)
    continuous = np.array(continuous)

    # Calculate means for the two groups
    mean_1 = continuous[binary == 1].mean()
    mean_0 = continuous[binary == 0].mean()
    
    # Standard deviation of the continuous variable
    std_dev = continuous.std()
    
    # Proportions and counts
    n_1 = (binary == 1).sum()
    n_0 = (binary == 0).sum()
    n = len(binary)

    # Biserial correlation formula
    r_b = (mean_1 - mean_0) / std_dev * np.sqrt((n_1 * n_0) / (n**2))
    
    return r_b

r_b = biserial_correlation(df['truth_value'], df['log_word_count'])
print("Biserial Correlation of Log Word Count:", r_b)

r_b = biserial_correlation(df['truth_value'], df['excl_word_count'])
print("Biserial Correlation of Exclusive Word Count:", r_b)

r_b = biserial_correlation(df['truth_value'], df['pers_pro_count'])
print("Biserial Correlation of Personal Pronoun Count:", r_b)

r_b = biserial_correlation(df['truth_value'], df['neg_word_count'])
print("Biserial Correlation of Negative Word Count:", r_b)

r_b = biserial_correlation(df['truth_value'], df['lexical_diversity'])
print("Biserial Correlation of Lexical Diversity:", r_b)

#%%
# Randomly split data into testing and training data

df = pd.read_csv('Processed Reviews with Features.csv')
testing = pd.read_csv('Testing Reviews.csv')
training = pd.read_csv('Training Reviews.csv')

testing = df.sample(frac=0.2, random_state=54)
training = df.drop(testing.index)

#%%
# Calculate mutual information of features

mi_word_count = mutual_info_score(training['truth_value'], training['log_word_count'])
print("Mutual Information of Word Count and Truth-value:", mi_word_count)

mi_excl_word_count = mutual_info_score(training['truth_value'],
                                       training['excl_word_count'])
print("Mutual Information of Exclusive Word Count and Truth-value:",
      mi_excl_word_count)

mi_prps_tag_count = mutual_info_score(training['truth_value'], training['pers_pro_count'])
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

X_train = training[['log_word_count']]
X_test = testing[['log_word_count']]
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


X_train = training[['pers_pro_count']]
X_test = testing[['pers_pro_count']]

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

probs = pd.DataFrame({"word_count": word_count_deception_probs, "excl_word_count": excl_word_count_deception_probs, "pers_pro_count": prps_tag_count_deception_probs, "neg_word_count": neg_word_count_deception_probs, "lexical_diversity": lex_div_deception_probs})

# calculate final probabilities

final_probs = np.dot(probs, measure)
testing['final_probs'] = final_probs
testing['guess'] = (testing['final_probs'] < 0.5).astype(int)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(testing['truth_value'], testing['guess'])
precision = precision_score(testing['truth_value'], testing['guess'])
recall = recall_score(testing['truth_value'], testing['guess'])
f1= f1_score(testing['truth_value'], testing['guess'])

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")
