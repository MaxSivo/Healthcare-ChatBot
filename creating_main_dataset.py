import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Reading Data
df = pd.read_csv("dataset.csv")
df_descriptions = pd.read_csv("descriptions.csv")
df_recommendations = pd.read_csv("recommendations.csv")

# Creating a column with a list of all the present symptoms
entries = df.shape[0]
for i in range(entries):
    values = df.iloc[i].values
    values = values.tolist()
    values = [x for x in values if str(x) != 'nan']
    values.pop(0)
    df['all_symptoms'][i] = values

# Creating a list of unique symptoms
symptoms_list = df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
                    'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9',
                    'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14',
                    'Symptom_15', 'Symptom_16', 'Symptom_17']].values.ravel()
df_symptoms = pd.unique(symptoms_list)
symptoms = df_symptoms.tolist()
symptoms = [i for i in symptoms if str(i) != "nan"]

# Creating a new dataframe with unique symptoms as column headers
df_symptoms = pd.DataFrame(columns=symptoms, index=df.index)
df_symptoms['all_symptoms'] = df['all_symptoms']

# Converting table values to numeric
for i in symptoms:
    df_symptoms[i] = df_symptoms.apply(lambda x: 1 if i in x.all_symptoms else 0, axis=1)

# Dropping 'all symptoms' column and adding the disease column
df_symptoms['Disease'] = df['Disease']
df_symptoms = symptoms.drop('all_symptoms', axis=1)

# Creating training and testing datasets
train, test = train_test_split(df_symptoms, test_size=0.2)
X_train = train.drop("Disease", axis=1)
y_train = train["Disease"].copy()
X_test = test.drop("Disease", axis=1)
y_test = test["Disease"].copy()

print(symptoms)
