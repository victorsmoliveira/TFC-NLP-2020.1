# %%imports & func
import os
import re
from warnings import simplefilter

import numpy as np
import pandas as pd
import spacy
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import KFold
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset


def clean_text(msg):
    msg = re.sub("[^a-zA-Z]", " ", msg)
    msg.lower()
    return msg


def lemmatization(msg):
    msg = clean_text(msg)
    doc = nlp(msg)
    tokenlist = [token.lemma_ for token in doc if token not in nlp.Defaults.stop_words]
    msg = " ".join(tokenlist)
    return msg


# %% Prep
simplefilter("ignore", category=ConvergenceWarning)
nlp = spacy.load("en_core_web_sm")
# %%Importing parameters & loading dataset
df_param = pd.read_csv(os.path.join("data", "params.csv"), delimiter=";")

dataset = pd.read_excel(os.path.join("data", "dataset_onehot.xlsx"))
labels_dataframe = dataset.iloc[:, 1:]

corpus = [lemmatization(msg) for msg in list(dataset["msgContent"])]

# %%main
y = labels_dataframe.values
lines, _ = df_param.shape

m_acc_array = []
m_hamm_loss_array = []
indexes = np.array([i for i in range(lines)])

for i in indexes:
    method = eval(df_param.iloc[i, 2])
    max_ft = df_param.iloc[i, 3]
    classifier = eval(df_param.iloc[i, 4])
    max_iterations = df_param.iloc[i, 5]
    k = df_param.iloc[i, 6]

    cv = CountVectorizer(max_features=max_ft)
    X = cv.fit_transform(corpus)

    acc_array = []
    hamm_loss_array = []

    kf = KFold(n_splits=k)

    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = (
            X[train_index],
            X[test_index],
            y[train_index],
            y[test_index],
        )

        cf = method(classifier(max_iter=max_iterations, random_state=42))
        cf.fit(X_train, y_train)
        y_pred = cf.predict(X_test)

        acc_array.append(accuracy_score(y_test, y_pred))
        hamm_loss_array.append(hamming_loss(y_test, y_pred))

    acc = np.mean(acc_array)
    hamm_loss = np.mean(hamm_loss_array)

    m_acc_array.append(acc)
    m_hamm_loss_array.append(hamm_loss)

srs_indexes = pd.Series(indexes + 1, name="Test")
df_results = pd.DataFrame({"Accuracy": m_acc_array, "Hamming Loss": m_hamm_loss_array})

df_complete = pd.concat([srs_indexes, df_param, df_results], axis=1)

df_complete.to_excel(
    os.path.join("resultados.xlsx"),
    index=False,
    freeze_panes=(1, 0),
    float_format="%.6f",
)
