import math
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize
import seaborn as sns
sns.set_theme()
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Embedding
from keras.utils import np_utils, pad_sequences
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.layers import SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

sample = pd.read_csv('Data/sample_submission/sample_submission.csv')
test = pd.read_csv('Data/test/test.csv')
train = pd.read_csv('Data/train/train.csv')
output_file_path = 'test_output.txt'

def multiclass_log_loss(y_true_binarized, y_pred_probabilities, epsilon=1e-20):
    """
    Computes the multiclass version oLf Log Loss metric
    Args:
        y_true_binarized (array_like, shape (m, n_class)): Matrix of binarized true target classes
        y_pred_probabilities (array_like, shape (m, n_class)): Matrix of predicted class probabilities
        eps (scalar): Clipping parameter for predicted class probabilities
            Class probabilities outside the interval (eps, 1 - eps) are clipped to the nearest endpoint
    Returns:
        logloss (scalar): Multiclass log loss obtained from y_true and y_pred (eps-clipped)
    """
    # Clipping to avoid undefined quantity log 0
    y_pred_probabilities = K.clip(y_pred_probabilities, epsilon, 1 - epsilon)

    # Casting sum of the elements of Hadamard product in (3)
    sum_ = tf.cast(K.sum(y_true_binarized * K.log(y_pred_probabilities)), tf.float64)

    # Computing log loss
    logloss = (-1 / len(y_true_binarized)) * sum_

    return logloss

def clean_stopwords(text):
    stop_words = nltk.corpus.stopwords.words('english')
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_text)
    return filtered_text

def clean_non_words(text):
    filtered_text = re.sub(r'\W+', ' ', text)
    return filtered_text

print("Before: ", train['text'].head())
train['text'] = train['text'].apply(clean_non_words)
train['text'] = train['text'].apply(clean_stopwords)
print("After: ", train['text'].head())

x, y = train['text'].values, train['author'].values
x_train, x_valid, y_train, y_valid = train_test_split(x, y,
                                                      stratify = y,
                                                      random_state = 40,
                                                      test_size = 0.2,
                                                      shuffle = True)
label_dict = {'EAP': 0, 'HPL': 1, 'MWS': 2}
y_train = pd.Series(y_train).replace(label_dict, inplace = False).values
y_valid = pd.Series(y_valid).replace(label_dict, inplace = False).values


def multiclass_log_loss(y_true_binarized, y_pred_probabilities, epsilon=1e-20):
    """
    Computes the multiclass version oLf Log Loss metric
    Args:
        y_true_binarized (array_like, shape (m, n_class)): Matrix of binarized true target classes
        y_pred_probabilities (array_like, shape (m, n_class)): Matrix of predicted class probabilities
        eps (scalar): Clipping parameter for predicted class probabilities
            Class probabilities outside the interval (eps, 1 - eps) are clipped to the nearest endpoint
    Returns:
        logloss (scalar): Multiclass log loss obtained from y_true and y_pred (eps-clipped)
    """
    # Clipping to avoid undefined quantity log 0
    y_pred_probabilities = K.clip(y_pred_probabilities, epsilon, 1 - epsilon)

    # Casting sum of the elements of Hadamard product in (3)
    sum_ = tf.cast(K.sum(y_true_binarized * K.log(y_pred_probabilities)), tf.float64)

    # Computing log loss
    logloss = (-1 / len(y_true_binarized)) * sum_

    return logloss

glove_6b_100d = 'Data/glove.6B.100d.txt'

def load_glove_model(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = set()
        word_vectors = {}
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = [float(val) for val in parts[1:]]
            words.add(word)
            word_vectors[word] = vector
    return words, word_vectors

word_set, glove_vectors = load_glove_model(glove_6b_100d)
print(f"Number of word vectors: {len(glove_vectors)}")
print("Word: 'the'")
print("Vector representation:")
print(glove_vectors['the'])

y_train_matrix = np_utils.to_categorical(y_train)
y_valid_matrix = np_utils.to_categorical(y_valid)

token = text.Tokenizer(num_words = None)
token.fit_on_texts(list(x_train) + list(x_valid))
word_index = token.word_index

x_train_seq = token.texts_to_sequences(x_train)
x_valid_seq = token.texts_to_sequences(x_valid)

# Example
print(f"Text: {x_train[0]}")
print("Converted to sequence:")
print(x_train_seq[0])

len_train = [len(x_train_seq[i]) for i in range(len(x_train_seq))]
len_valid = [len(x_valid_seq[i]) for i in range(len(x_valid_seq))]
len_ = np.array(len_train + len_valid)
maxlen_ = math.floor(len_.mean() + 2*len_.std()) + 1

x_train_pad = pad_sequences(x_train_seq,
                                    maxlen = maxlen_,
                                    padding = 'pre',
                                    truncating = 'pre',
                                    value = 0.0)

x_valid_pad = pad_sequences(x_valid_seq,
                                    maxlen = maxlen_,
                                    padding = 'pre',
                                    truncating = 'pre',
                                    value = 0.0)
dim_glove = len(glove_vectors['the'])

word_vectorization_matrix = np.zeros((len(word_index) + 1, dim_glove))
for word, i in word_index.items():
    word_embed_vector = glove_vectors.get(word)
    if word_embed_vector is not None:
        word_vectorization_matrix[i] = word_embed_vector

print(f"Shape of the matrix of word vectors: {word_vectorization_matrix.shape}")

# LSTM with GloVe embeddings
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                    dim_glove,
                    weights = [word_vectorization_matrix],
                    input_length = maxlen_,
                    trainable = False))

model.add(SpatialDropout1D(0.3))
model.add(LSTM(dim_glove, dropout = 0.3, recurrent_dropout = 0.3))

model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.8))

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.8))

model.add(Dense(3, activation = 'softmax'))

print(model.summary())
initial_learning_rate = 0.001
model.compile(loss = multiclass_log_loss,
              optimizer = tf.keras.optimizers.Adam(learning_rate = initial_learning_rate))
# Early stopping callback
earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0.001,
                          patience = 20,
                          verbose = 1,
                          mode = 'auto')


# Modified linear schedule function
def scheduler_modified_exponential(epoch, learning_rate):
    if epoch < 40:
        return learning_rate
    else:
        return learning_rate * math.exp(-0.1)


learning_rate, epoch, num_epochs, learning_rate_list = initial_learning_rate, 0, 100, []
for i in range(num_epochs):
    learning_rate = scheduler_modified_exponential(epoch, learning_rate)
    learning_rate_list.append(learning_rate)
    epoch += 1
plt.figure(figsize=(15, 10))
plt.plot(learning_rate_list)
plt.title("Learning rate vs Epoch", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Learning rate", fontsize=14)
plt.savefig('learningrate_modified_expoential.jpg')
plt.show()

learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler_modified_exponential)
# Model fitting
history = model.fit(x_train_pad,
                    y = y_train_matrix,
                    batch_size = 256,
                    epochs = 100,
                    verbose = 0,
                    validation_data = (x_valid_pad, y_valid_matrix),
                    callbacks = [earlystop, learning_rate_scheduler])

# Visualization of model loss
plt.figure(figsize = (15, 10))
plt.title('Model loss (multiclass log loss)', fontsize = 14)
sns.lineplot(data = history.history['loss'], label = 'Train')
sns.lineplot(data = history.history['val_loss'], label = 'Validation')
plt.xlabel('Epoch', fontsize = 14)
plt.ylabel('Loss', fontsize = 14)
plt.legend()
plt.savefig('logloss.jpg')
plt.show()

y_train_matrix_pred = model.predict(x_train_pad)
y_valid_matrix_pred = model.predict(x_valid_pad)

# Log loss
logloss_train = multiclass_log_loss(y_train_matrix, y_train_matrix_pred).numpy()
logloss_valid = multiclass_log_loss(y_valid_matrix, y_valid_matrix_pred).numpy()

# Log loss (computed using model.evaluate)
# logloss_train = model.evaluate(x_train_pad, y_train_matrix, batch_size = 256, verbose = 0)
# logloss_valid = model.evaluate(x_valid_pad, y_valid_matrix, batch_size = 256, verbose = 0)

print(f"Training logloss  : {round(logloss_train, 3)}")
print(f"Validation logloss: {round(logloss_valid, 3)}")

# Converting probability vectors to labels
y_train_true = np.array([np.argmax(x) for x in y_train_matrix])
y_valid_true = np.array([np.argmax(x) for x in y_valid_matrix])
y_train_pred = np.array([np.argmax(x) for x in y_train_matrix_pred])
y_valid_pred = np.array([np.argmax(x) for x in y_valid_matrix_pred])
# Accuracy
match_train = (y_train_true == y_train_pred)
match_train = np.array(list(map(lambda x: int(x == True), match_train)))
match_valid = (y_valid_true == y_valid_pred)
match_valid = np.array(list(map(lambda x: int(x == True), match_valid)))

print(f"Training accuracy  : {round(match_train.sum() / len(match_train), 3)}")
print(f"Validation accuracy: {round(match_valid.sum() / len(match_valid), 3)}")

def conf_matrix(y_true, y_pred, n_class, class_names = 'default', figsize = (6.25, 5), font_scale = 1, annot_kws_size = 12):
    if class_names == 'default':
        class_names = np.arange(n_class)
    tick_marks_y = np.arange(n_class) + 0.5
    tick_marks_x = np.arange(n_class) + 0.5
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    confusion_matrix_df = pd.DataFrame(confusion_matrix, range(n_class), range(n_class))
    plt.figure(figsize = figsize)
    sns.set(font_scale = font_scale) # label size
    plt.title("Confusion Matrix")
    sns.heatmap(confusion_matrix_df, annot = True, annot_kws = {"size": annot_kws_size}, fmt = 'd') # font size
    plt.yticks(tick_marks_y, class_names, rotation = 'vertical')
    plt.xticks(tick_marks_x, class_names, rotation = 'horizontal')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.savefig('confmat.jpg')
    plt.show()

conf_matrix(y_valid_true, y_valid_pred, n_class = 3, class_names = ['EAP', 'HPL', 'MWS'])

