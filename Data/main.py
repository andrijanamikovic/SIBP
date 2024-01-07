import base64
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import seaborn as sns



sample = pd.read_csv('sample_submission/sample_submission.csv')
test = pd.read_csv('test/test.csv')
train = pd.read_csv('train/train.csv')

import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')


print(train.head())
print(test.head())
print(sample.head())

colors = ['dodgerblue', 'lightcoral', 'lightgreen']
author_mapping = {'EAP': 'Edgar Allen Poe', 'MWS': 'Mary Shelley', 'HPL': 'HP Lovecraft'}
author_counts = train['author'].map(author_mapping).value_counts()
plt.bar(author_counts.index, author_counts.values, color=colors)
plt.xlabel('Author')
plt.ylabel('Count')
plt.title('Target variable distribution')
# plt.show()

def plot_most_used_words(data_to_plot, label):
    plt.figure(figsize=(10, 6))
    plt.bar(data_to_plot.index.values[2:50], data_to_plot.values[2:50], color=colors)
    plt.xlabel('Words')
    plt.ylabel('Word Counts')
    plt.title(label)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()

#Which word are most used, after removing stop_words
def get_word_counts(texts):
    stop_words = set(stopwords.words('english'))
    words = ' '.join(texts).lower()  # Combine all texts and convert to lowercase
    words = word_tokenize(words)  # Tokenize the text
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]  # Filter out stop words and non-alphabetic words
    word_counts = pd.Series(filtered_words).value_counts()
    return word_counts

all_words = get_word_counts(train['text'])

# Word frequency counts for each author

eap = get_word_counts(train[train.author == "EAP"]['text'])
hpl = get_word_counts(train[train.author == "HPL"]['text'])
mws = get_word_counts(train[train.author == "MWS"]['text'])


plot_most_used_words(all_words, 'Top 50  Word frequencies in the training dataset')
plot_most_used_words(eap, 'Top 50  Word frequencies used by Edgar Allen Poe in  the training dataset')
plot_most_used_words(hpl, 'Top 50  Word frequencies used by HP Lovecraft in the training dataset')
plot_most_used_words(mws, 'Top 50  Word frequencies used by Mary Shelley in the training dataset')
# plt.show()

## Number of words in the text ##
train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train["num_chars"] = train["text"].apply(lambda x: len(str(x)))
test["num_chars"] = test["text"].apply(lambda x: len(str(x)))

## Number of punctuations in the text ##
train["num_punctuations"] =train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["num_punctuations"] =test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train["num_words_upper"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["num_words_upper"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train["num_words_title"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test["num_words_title"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train["mean_word_len"] = train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["mean_word_len"] = test["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

train['num_words'].loc[train['num_words']>80] = 80 #truncation for better visuals
plt.figure(figsize=(12,8))
sns.violinplot(x='author', y='num_words', data=train)
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Number of words in text', fontsize=12)
plt.title("Number of words by author", fontsize=15)

train['num_punctuations'].loc[train['num_punctuations']>10] = 10 #truncation for better visuals
plt.figure(figsize=(12,8))
sns.violinplot(x='author', y='num_punctuations', data=train)
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Number of puntuations in text', fontsize=12)
plt.title("Number of punctuations by author", fontsize=15)
plt.show()

#clean up and tokenization
first_text = train.text.values[0]
print(first_text)
first_text_list = word_tokenize(first_text)
print(first_text_list)
stop_words = nltk.corpus.stopwords.words('english')
first_text_list_cleaned = [word for word in first_text_list if word.lower() not in stop_words]
print(first_text_list_cleaned)
print("="*90)
print("Length of original list: {0} words\n"
      "Length of list after stopwords removal: {1} words"
      .format(len(first_text_list), len(first_text_list_cleaned)))
print(first_text_list_cleaned)

#stemmer ne koristim ovo imam primer sto
# stemmer = nltk.stem.PorterStemmer()
# first_text_list_cleaned = [stemmer.stem(word) for word in first_text_list_cleaned]
# print("="*90)
# print(first_text_list_cleaned)
#Lemmatization
lemm = WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))


# Define helper function to print top words
def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
        print("="*70)


# Storing the entire training text in a list
text = list(train.text.values)
# Calling our overwritten Count vectorizer
tf_vectorizer = LemmaCountVectorizer(max_df=0.95,
                                     min_df=2,
                                     stop_words='english',
                                     decode_error='ignore')
tf = tf_vectorizer.fit_transform(text)
feature_names = tf_vectorizer.get_feature_names_out()
count_vec = np.asarray(tf.sum(axis=0)).ravel()
data_to_plot = pd.Series(count_vec, index=feature_names)

# Sort data_to_plot by word frequencies
data_to_plot = data_to_plot.sort_values(ascending=False)
plot_most_used_words(data_to_plot, 'Top 50  Word frequencies after preprocessing')

# plt.show()

#LatentDirichletAllocation
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author.values)
