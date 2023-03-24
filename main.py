import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# Telling user to ignore setting with copy warning
print('\nTraining machine learning, your chatbot will be ready soon.')
print('Setting with copy warning is not an error, please wait.\n')

# Reading Data
df = pd.read_csv("dataset.csv")
df_desc = pd.read_csv("descriptions.csv")
df_recs = pd.read_csv("recommendations.csv")


# Fixing spelling mistake found in description dataset
df_desc.iloc[16, 0] = 'Dimorphic hemmorhoids(piles)'


# Creating dictionary of disease descriptions and of recommendations
desc_dict = df_desc.set_index('Disease').T.to_dict('list')
rec_dict = df_recs.set_index('Disease').T.to_dict('list')


# Creating a column with a list of all the present symptoms
df['all_symptoms'] = 0
entries = df.shape[0]

for i in range(entries):
    values = df.iloc[i].values
    values = [x for x in values if str(x) != 'nan' and x != 0]
    values = [x.replace('_', ' ') for x in values]
    values = [x.strip() for x in values]
    values = [x.replace(' ', '') for x in values]
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


# clean symptoms for user-friendliness
cleaned_symptoms_user = sorted(list(map(lambda x: x.replace('_', ' '), symptoms)))
cleaned_symptoms_user = [x.strip() for x in cleaned_symptoms_user]
cleaned_symptoms = [x.replace(' ', '') for x in cleaned_symptoms_user]


# Creating a new dataframe with unique symptoms as column headers
df_symptoms = pd.DataFrame(columns=cleaned_symptoms, index=df.index)
df_symptoms['all_symptoms'] = df['all_symptoms']


# Converting table values to numeric
for i in cleaned_symptoms:
    df_symptoms[i] = df_symptoms.apply(lambda x: 1 if i in x.all_symptoms else 0, axis=1)


# Dropping 'all symptoms' column and adding the disease column
df_symptoms['Disease'] = df['Disease']
df_symptoms = df_symptoms.drop('all_symptoms', axis=1)


# Creating target variable and training dataframe
X = df_symptoms.drop("Disease", axis=1)
y = df_symptoms["Disease"]


# Creating random forest model and checking accuracy with cross validation
rfc = RandomForestClassifier()
cv = cross_val_score(rfc, X, y, cv=5)


# checking if model scored 100% accuracy
if cv.mean() == 1:
    print('Training successful.\n')


# Fitting model
rfc.fit(X, y)


def greeting():
    print('\nHi! I am a chatbot designed to diagnose your illness and provide you with recommendations.')
    username = input('Please enter your name: ')
    print(f'Hello, {username}!\n')
    get_symptoms()


def get_symptoms():
    symptom_count = 0

    # Handling invalid symptom number
    while True:
        try:
            symptom_count += int(input('Please enter the number of symptoms you are experiencing (max 10): '))
        except ValueError:
            print('Please enter a valid number. \n')
            continue
        else:
            break

    print(f'Here are the symptoms we accept: {cleaned_symptoms_user} \n')

    # Prompting symptoms
    user_symptoms = []
    while True:
        symp = input(f'Enter symptom {len(user_symptoms) + 1}: ').lower()
        symp = symp.replace(' ', '')
        if symp in cleaned_symptoms:
            user_symptoms.append(symp)
        elif symp in user_symptoms:
            print('You have already entered that symptom.')
        else:
            print('Invalid symptom, please see valid symptoms list.')
        if symptom_count == len(user_symptoms):
            break

    get_prognosis(user_symptoms=user_symptoms)


def get_prognosis(user_symptoms):
    # Creating user dataframe with user symptoms
    df_user = pd.DataFrame(columns=cleaned_symptoms)
    df_user.loc[1] = 0
    for s in user_symptoms:
        df_user[s] = 1

    # Predicting disease based on symptoms
    disease_pred = rfc.predict(df_user)
    disease = ''.join(disease_pred)
    print(f'\nIt appears you have the disease {disease}:')

    # Getting disease description
    description = ''.join(desc_dict[disease])
    print(f'{description}\n')

    # Getting recommendations
    print('Here are the recommended actions you follow:\n')
    for item in rec_dict[disease]:
        if str(item) != 'nan':
            print(f'{item}\n')


greeting()
