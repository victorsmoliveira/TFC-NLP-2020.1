# %%imports & func
import pandas as pd
import re
import spacy
import numpy as np   
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, accuracy_score

def clean_text(msg):
    msg = re.sub('[^a-zA-Z]', ' ', msg)
    msg.lower()
    return msg

def lemmatization(msg):
    msg = clean_text(msg)
    doc = nlp(msg)
    tokenlist = [token.lemma_ for token in doc if token not in nlp.Defaults.stop_words]
    msg = ' '.join(tokenlist)
    return msg

nlp = spacy.load('en_core_web_sm')
# %%Importing parameters & loading dataset

script_path = os.path.dirname(__file__)

df_param = pd.read_csv(os.path.join(script_path, 'Parametros.csv'), delimiter = ';')

dataset = pd.read_excel(os.path.join(script_path, 'Datasets\Dataset OneHot.xlsx'))
labels_dataframe = dataset.iloc[:, 1:]

corpus = [lemmatization(msg) for msg in list(dataset['msgContent'])]

# %%main
y = labels_dataframe.values
lines,_ = df_param.shape

m_acc_array = []
m_hamm_loss_array = []
indexes = np.array([i for i in range(lines)])

for i in indexes:
    method = eval(df_param.iloc[i,2])
    max_ft = df_param.iloc[i,3]
    classifier = eval(df_param.iloc[i,4])
    max_iterations = df_param.iloc[i,5]
    k = df_param.iloc[i,6]

    cv = CountVectorizer(max_features=max_ft)
    X = cv.fit_transform(corpus)

    acc_array = []
    hamm_loss_array = []

    kf = KFold(n_splits=k)

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

    m_acc_array.append(acc)
    m_hamm_loss_array.append(hamm_loss)

df_excel = pd.DataFrame({'Test':indexes+1,'Accuracy':m_acc_array,'Hamming Loss':
                         m_hamm_loss_array})
    
df_excel.to_excel(os.path.join(script_path, 'Resultados.xlsx'), index=False)

