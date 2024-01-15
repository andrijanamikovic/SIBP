import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
sns.set_theme()
from sklearn.model_selection import train_test_split
import warnings

sample = pd.read_csv('Data/sample_submission/sample_submission.csv')
test = pd.read_csv('Data/test/test.csv')
train = pd.read_csv('Data/train/train.csv')
warnings.filterwarnings("ignore", category=FutureWarning)

import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('omw-1.4')

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
plt.savefig('target_variable_distribution.jpg')

# plt.show()

def plot_most_used_words(data_to_plot, label):
    plt.figure(figsize=(10, 6))
    plt.bar(data_to_plot.index.values[2:50], data_to_plot.values[2:50], color=colors)
    plt.xlabel('Words')
    plt.ylabel('Word Counts')
    plt.title(label)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.savefig(f'{label}.jpg')


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

document_lengths = np.array(list(map(len, train.text.str.split(' '))))

print("The average number of words in a document is: {}.".format(np.mean(document_lengths)))
print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
print("The maximum number of words in a document is: {}.".format(max(document_lengths)))

fig, ax = plt.subplots(figsize=(15,6))
ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
sns.histplot(document_lengths, bins=50, ax=ax)
plt.savefig('distribution_of_number_of_words.jpg')


# plt.show()


## Prepare the data for modeling ###
author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
train['author'] = train['author'].map(author_mapping_dict)
train_id = train['id'].values
test_id = test['id'].values

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


###


texts = train.text

#Lemmatization
lemm = WordNetLemmatizer()
def lemamatization_process(text):
    words = word_tokenize(text)
    lemmatized_words = [lemm.lemmatize(word.lower()) for word in words]
    lematized_text = " ".join(lemmatized_words)
    return lematized_text

def clean_text(text):
    first_text_list = word_tokenize(text)
    stop_words = nltk.corpus.stopwords.words('english')
    first_text_list_cleaned = ' '.join(word for word in first_text_list if word.lower() not in stop_words)
    first_text_list_cleaned = re.sub(r'\W',' ', first_text_list_cleaned)
    return first_text_list_cleaned

train['cleaned_text'] = train['text'].apply(lemamatization_process)
train['cleaned_text'] = train['cleaned_text'].apply(clean_text)
print(train[['text', 'cleaned_text']].head())
X = train
X = X.drop('text', axis=1)
X = X.drop('id', axis=1)


X_train_part, X_test, Y_train, Y_test =\
    train_test_split(X.drop('author', axis=1), X['author'], test_size=0.3,random_state=17, stratify=train['author'])


document_train = X_train_part['cleaned_text'].tolist()
document_test = X_test['cleaned_text'].tolist()
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(document_train)
X_test = tfidf.transform(document_test)

def train_model(model, label):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(Y_test, Y_pred), 3)
    precision = round(precision_score(Y_test, Y_pred, average='weighted'), 5)
    recall = round(recall_score(Y_test, Y_pred, average='weighted'), 5)
    fscore = round(f1_score(Y_test, Y_pred, average='weighted'), 5)
    print(f"Accuracy of the model: {np.round(accuracy*100, 5)}%")
    print(f"Precision of the model: {np.round(precision*100, 5)}%")
    print(f"Recall of the model: {np.round(recall*100, 5)}%")
    print(f"F1 score of the model: {np.round(fscore*100, 5)}%")

    # Plot Confusion Matrix
    disp = plot_confusion_matrix(model, X_test, Y_test, cmap=plt.cm.Blues, values_format='d', display_labels=np.unique(Y_test))
    disp.ax_.set_title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_{label}.jpg')

def knn_algotiham_plot():
    knn_score_list = []
    for k in range(2, 50, 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)  # ubaci u model
        prediction = knn.predict(X_test)
        knn_score_list.append(knn.score(X_test, Y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 50, 1), knn_score_list, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='green', markersize=10)
    plt.title('Precision of KNN for different k values')
    plt.xlabel('K')
    plt.ylabel('Precision')
    plt.savefig('knn_alg_values.jpg')

# knn_algotiham_plot()
print("KNN:")
train_model(KNeighborsClassifier(n_neighbors=43), 'KNN')
print("LogisticRegression:")
train_model(LogisticRegression(max_iter=2000), "logistic_regression")
print("DecisionTree")
train_model(DecisionTreeClassifier(), 'decision_tree')
print("MultinomialNB")
train_model(MultinomialNB(), 'multinomialNB')
