# %%imports & func
import os
import pandas as pd
import re
import spacy
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss, accuracy_score


def clean_text(msg):
    msg = msg.lower()
    msg = re.sub('[^a-zA-Z]', ' ', msg)
    msg = ' '.join(msg.split())
    return msg


def stemming(msg):
    msg = clean_text(msg)
    stemmer = SnowballStemmer(language='english')
    wordlist = [stemmer.stem(word) for word in msg.split() if word not in
                set(stopwords.words('english'))]
    msg = ' '.join(wordlist)
    return msg


def lemmatization(msg):
    msg = clean_text(msg)
    doc = nlp(msg)
    tokenlist = [token.lemma_ for token in doc if str(token) not in
                 set(nlp.Defaults.stop_words)]
    msg = ' '.join(tokenlist)
    return msg


def word_freq(wordlist, n):
    big_sen = ''

    for sen in wordlist:
        big_sen += sen

    return Counter(big_sen.split()).most_common(n)


# %%Parameters part 1
norm = lemmatization  # stemming, lemmatization
stopwords_cond = True  # True, False

# %%Adding stopwords
nlp = spacy.load('en_core_web_sm')

if stopwords_cond:
    for word in nlp.Defaults.stop_words:
        lex = nlp.vocab[word]
        lex.is_stop = True

    stopwords.words('english').extend(['a', 'e', 'n', 's', 'm', 'o', 'br', 'ls',
                                       'www', 'com', 'fot', 'cs', 'hayasa', 'apollo'])

    nlp.Defaults.stop_words |= {'a', 'e', 'n', 's', 'm', 'o', 'br', 'ls', 'www',
                                'com', 'fot', 'cs', 'hayasa', 'apollo'}

# %%Loading dataset
dataset = pd.read_excel('Datasets/Dataset OneHot.xlsx')
labels_dataframe = dataset.iloc[:, 1:]

# %%Visualizing classes
import seaborn as sns

sns.set()
sns.set_style('darkgrid')
plt.subplots(figsize=(10, 9))

sum_msgs = labels_dataframe.sum(axis=0).sort_values(ascending=False)

ax = sns.barplot(y=labels_dataframe.columns.values, x=sum_msgs, color='#B1587E')

for i, val in enumerate(sum_msgs):
    ax.text(val + 30, i, val, color='black', va='center', fontsize=10)

plt.ylabel('Ato de Fala', fontsize=13)
plt.xlabel('Contagem', fontsize=13)
plt.title('Contagem de mensagens por Ato de Fala', fontsize=16)

plt.savefig('cont msgs.png', dpi=300)

# %%Visualizing classes
sns.set_style('white')

count_lbl = (pd.DataFrame(labels_dataframe
                          .sum(axis=1)
                          .sort_values(ascending=False))
             .reset_index()
             .groupby(0).count()
             .reset_index()
             .rename(columns={0: 'num_lbl', 'index': 'count'}))

plt.subplots(figsize=(10, 9))
ax = sns.barplot(x=count_lbl['num_lbl'],
                 y=count_lbl['count'],
                 color='#833788')

for i, num_lbl, count in count_lbl.itertuples():
    ax.text(i, count + 30, count, color='black', ha='center', fontsize=11)

plt.xlabel('Quantidade de labels na mensagem', fontsize=13)
plt.ylabel('Número de mensagens', fontsize=13)
plt.title('Número de mensagens por quantidade de labels', fontsize=16)

# plt.savefig('count lbl.png', dpi=300)
# %%Visualizing classes
sns.set_style('ticks')

len_msgs = [len(msg) for msg in dataset['msgContent']]

plt.subplots(figsize=(10, 9))
sns.distplot(len_msgs, hist=False, color='black')
plt.xlabel('Tamanho da mensagem (quantidade de caracteres)', fontsize=13)
plt.title('Histograma de frequencia do tamanho das mensagens', fontsize=16)

plt.savefig('hist_freq.png', dpi=300)
# %%Visualizing classes

len_msgs = [len(msg) for msg in dataset['msgContent'] if len(msg) <= 500]

plt.subplots(figsize=(10, 9))
sns.distplot(len_msgs, bins=20, kde=False, hist_kws=dict(alpha=1), color='#635751')
plt.xlabel('Tamanho da mensagem (quantidade de caracteres)', fontsize=13)
plt.title('Histograma do tamanho das mensagens com tamanho menor ou igual a 500',
          fontsize=16)

plt.savefig('hist_tam_msgs.png', dpi=300)
# %%Corpus creation
corpus = [norm(msg) for msg in list(dataset['msgContent'])]

# %%Parameters part 2
method = BinaryRelevance  # BinaryRelevance, LabelPowerset
classifier = LogisticRegression  # LogisticRegression, SVC, GaussianNB
max_iterations = 40
max_ft = 2000
k = 5

# %%Creating model & splitting dataset

# Creating the Bag of Words model
cv = CountVectorizer(max_features=max_ft)
X = cv.fit_transform(corpus)

y = labels_dataframe.values

acc_array = []
hamm_loss_array = []

# Splitting the dataset into training and test sets
kf = KFold(n_splits=k)

for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], \
                                       y[train_index], y[test_index]

    cf = method(classifier(max_iter=max_iterations, random_state=42))
    cf.fit(X_train, y_train)
    y_pred = cf.predict(X_test)

    acc_array.append(accuracy_score(y_test, y_pred))
    hamm_loss_array.append(hamming_loss(y_test, y_pred))

acc = np.mean(acc_array)
hamm_loss = np.mean(hamm_loss_array)

# -------


my_msg = dataset.iloc[4562]['msgContent']

my_msg_clean = clean_text(my_msg)
my_msg_stem = stemming(my_msg)
my_msg_lemma =  lemmatization(my_msg)

wordlist = [word for word in my_msg_clean.split() if word not in 
            set(stopwords.words('english'))]

msg_stpwd = ' '.join(wordlist)
    

doc = nlp(msg_stpwd)
tokenlist = [token.lemma_ for token in doc]
msg_lem = ' '.join(tokenlist)
    
my_msg = list([my_msg])
my_X = cv.transform(my_msg)


my_pred = cf.predict(my_X)

my_pred = my_pred.todense()


# Printing used parameters
print('\n\033[4mParameters:\033[0m\n')
print(f'Normalization: {norm.__name__}')
print(f'\nMethod: {method.__name__}')
print(f'Classifier: {classifier.__name__}')
print(f'Max iterations: {max_iterations}')
print(f'Max features: {max_ft}')
print(f'k: {k}')

# Printing results
print('\n\033[4mMetrics:\033[0m\n')
print(f'Accuracy: {acc:.2%}')
print(f'Hamming Loss: {hamm_loss:.2%}')
