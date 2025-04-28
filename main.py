import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.read_csv('dataset/LoanApprovalPrediction.csv')

# print(data.info())
# print(data.describe())
# print(data.head())

# objects = data.columns[data.dtypes == 'object']
# print(objects)
# print(len(objects))
'7 категориальных признаков'

'Удалим ненужные данные (ID)'
data.drop(['Loan_ID'], axis=1, inplace=True)
objects = data.columns[data.dtypes == 'object']

'Отобразим графики'
# plt.figure(figsize=(16, 8))
# index = 1
#
# for column in objects:
#     y = data[column].value_counts()
#     plt.subplot(2, 3, index)
#     sns.barplot(x=y.index, y=y)
#     index += 1
#
# plt.tight_layout()
# plt.show()

'Преобразуем категориальные данные в численный тип'
label_encoder = LabelEncoder()

for column in objects:
    data[column] = label_encoder.fit_transform(data[column])

    # if column == 'Gender':
    #     print(label_encoder.classes_)

'Проверим количество категориальных признаков'
# objects = data.columns[data.dtypes == 'object']
# print(len(objects))
'0 категориальных признаков'

'Проверим коррелируемость признаков'
# plt.figure(figsize=(12, 8))
# sns.heatmap(data.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
# plt.show()

'Посмотрим зависимость семейного положения от пола человека'
# axis = sns.catplot(x='Gender', y='Married', hue='Loan_Status', kind='bar', data=data)
# axis.set_xticklabels(['Female', 'Male'])
# plt.show()

# print(data.isna().sum())
'Есть пропущенные значения, заменим средним'

for column in data.columns:
    data[column] = data[column].fillna(data[column].mean())

# print(data.isna().sum())

'Разделяем данные'
X = data.drop(['Loan_Status'], axis=1)
y = data['Loan_Status']
# print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

'Обучаем 4 модели'
knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC()
lc = LogisticRegression(max_iter=500)
rfc = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)

'Сделаем прогнозы на обучающей выборке'
print('Обучающая выборка:')
for clf in (knn, svm, lc, rfc):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)

    print(f'{clf.__class__.__name__}: доля правильных ответов - {accuracy_score(y_train, y_pred)}')

print()
'Лучший accuracy у случайного леса - 0.98'

'Сделаем прогнозы на тестовой выборке'
print('Тестовая выборка')
for clf in (knn, svm, lc, rfc):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f'{clf.__class__.__name__}: доля правильных ответов - {accuracy_score(y_test, y_pred)}')

'Логистическая регрессия и случайный лес дают лучшие результаты'
