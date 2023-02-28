# %% Imports and functions
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
from tqdm import tqdm, trange


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

print("\n--------- STARTING SCRIPT ---------\n")

print("📥 Loading Spacy dictionary...")
nlp = spacy.load("en_core_web_sm")
print("Dictionary loaded sucessfully.")

# %% Importing parameters & loading dataset
print("\n📖 Reading parameters csv...")
df_param = pd.read_csv(os.path.join("data", "params.csv"), delimiter=";")
nrows = len(df_param)
print(f"Read successfully. Found {nrows} different parameter sets.")

print("\n📖 Reading dataset...")
dataset = pd.read_excel(os.path.join("data", "dataset_onehot.xlsx"))
labels_dataframe = dataset.iloc[:, 1:]
print("Read successfully.")

print("\n🔨 Lemmatizating messages...")
corpus = [lemmatization(msg) for msg in list(dataset["msgContent"])]
print("Lemmatization complete.")

# %% main
y = labels_dataframe.values

m_acc_array = []
m_hamm_loss_array = []

print(f"\nStarting main loop over {nrows} parameter sets:\n")
for i in trange(nrows, desc="⏳️ General progress", colour="#84e175"):
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

    for train_index, test_index in tqdm(
        kf.split(X),
        desc="   💪 KFold training for parameter set #" + str(i + 1),
        total=k,
        leave=False,
        colour="#35edfe",
    ):
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

srs_indexes = pd.Series(range(nrows), name="N")
df_results = pd.DataFrame({"Accuracy": m_acc_array, "Hamming Loss": m_hamm_loss_array})

df_complete = pd.concat([srs_indexes + 1, df_param, df_results], axis=1)

output_filename = "results"

# Export to csv
df_complete.to_csv(output_filename + ".csv", index=False)
print(f"\n📝 Results successfully exported to {output_filename}.csv.")

# Export to excel
df_complete.to_excel(output_filename + ".xlsx", index=False, freeze_panes=(1, 0))
print(f"\n📝 Results successfully exported to {output_filename}.xlsx.")
