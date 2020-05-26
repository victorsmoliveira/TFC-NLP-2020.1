# %%imports & func
import pandas as pd
import re
import spacy
import numpy as np

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
    tokenlist = [token.lemma_ for token in doc if token not in nlp.Defaults.stop_words]
    msg = ' '.join(tokenlist)
    return msg


def word_freq(wordlist, n):
    big_sen = ''

    for sen in wordlist:
        big_sen += sen

    return Counter(big_sen.split()).most_common(n)


# %%Parameters part 1
norm = lemmatization  # stemming, lemmatization
stopwords_cond = True # True, False

# %%Adding stopwords
nlp = spacy.load('en_core_web_sm')

if stopwords_cond:
    nlp.Defaults.stop_words |= {'a', 'e', 'n', 's', 'm', 'o', 'br', 'ls', 'www', 'com',
                                'fot', 'cs', 'hayasa', 'apollo',}

    for word in nlp.Defaults.stop_words:
        lex = nlp.vocab[word]
        lex.is_stop = True

    stopwords.words('english').extend(['a', 'e', 'n', 's', 'm', 'o', 'br', 'ls', 'www', 'com',
                                       'fot', 'cs', 'hayasa', 'apollo'])

# %%Loading dataset & corpus creation
dataset = pd.read_excel(r'E:\OneDrive\PUC\11 Período\TCC\Dataset\Excel\Dataset Original '
                        r'OneHot.xlsx')

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
y = dataset.iloc[:, 1:].values

acc_array = []
hamm_loss_array = []

# Splitting the dataset into training and test sets
kf = KFold(n_splits=k)

from sklearn.metrics import zero_one_loss

for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], \
        y[test_index]

    cf = method(classifier(max_iter=max_iterations, random_state=42))
    cf.fit(X_train, y_train)
    y_pred = cf.predict(X_test)

    acc_array.append(accuracy_score(y_test, y_pred))
    hamm_loss_array.append(hamming_loss(y_test, y_pred))
    
acc = np.mean(acc_array)
hamm_loss = np.mean(hamm_loss_array)

# my_msg = norm('i really dont understand this')
# my_msg = list([my_msg])
# my_X = cv.transform(my_msg)

# my_pred = cf.predict(my_X)

# my_pred = my_pred.todense()

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
