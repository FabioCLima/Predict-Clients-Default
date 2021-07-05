# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['figure.dpi']=400

SRC = os.path.abspath('.')
BASE = os.path.dirname(SRC)
DATA = os.path.join(BASE, 'data')
MODELS = os.path.join(BASE, 'models')
FIGS = os.path.join(BASE, 'figs')

# %%
#! Dados de entrada para a criação do modelo
input_file = 'Chapter_1_cleaned_data.csv'
file_path = os.path.join(DATA, input_file)
df = pd.read_csv(file_path)
df.head()
# %%
#! Proporção da classe positiva --> média de inadiplência
df['default payment next month'].mean()
# %%
#! Proporção de cada classe no dataset
df.groupby('default payment next month')['ID'].count()
# %%
df_new = df.drop(columns=['ID']).copy()
data = df_new.drop(columns=['default payment next month']).copy()
# %%
target = df_new['default payment next month']
# %%
target.value_counts().plot.barh()
_ = plt.title('Número de amostras por classe presente\n no target')
# %%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier
# %%
classifier = LogisticRegression(
    C=1.0,
    class_weight=None,
    dual=False,
    fit_intercept=True,
    intercept_scaling=1,
    max_iter=100,
    multi_class='auto',
    n_jobs=None,
    penalty='l2',
    random_state=123,
    solver='warn',
    tol=0.0001,
    verbose=0,
    warm_start=False
)
# %%
print(classifier)
# %%
classifier.C = 0.1
classifier.solver = 'liblinear'
classifier
# %%
X = df['EDUCATION'][0:10].values.reshape(-1, 1)
X
# %%
y = df['default payment next month'][0:10].values
y
# %%
classifier.fit(X, y)
# %%
new_X = df['EDUCATION'][10:20].values.reshape(-1, 1)
new_X
# %%
classifier.predict(new_X)
# %%
df['default payment next month'][10:20].values
# %%
target_test = df['default payment next month'][10:20].values
target_predicted = classifier.predict(new_X)
resposta = np.mean(target_test == target_predicted)
print(f"O modelo com apenas uma variável, previu a não inadiplência com\n {resposta*100:.2f}%")
# %%
np.random.seed(seed=1)
X = np.random.uniform(low=0.0, high=10.0, size=(1000,))
X[0:10]
# %%
#! y = ax + b + N(mi, sigma)
np.random.seed(seed=1)
slope = 0.25
intercept = -1.25
y = slope * X + np.random.normal(loc=0.0, scale=1.0, size=(1000,)) + intercept   
# %%
mpl.rcParams['figure.dpi']=400
plt.scatter(X, y, s=1)
# %%
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg
# %%
lin_reg.fit(X.reshape(-1, 1), y)
print(lin_reg.intercept_)
print(lin_reg.coef_)
# %%
y_pred = lin_reg.predict(X.reshape(-1, 1))
plt.scatter(X, y, s=1)
plt.plot(X, y_pred, 'r')
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['EDUCATION'].values.reshape(-1, 1), 
    df['default payment next month'].values,
    test_size=0.2, random_state=24,
    stratify=df['default payment next month'].values
)
# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# %%
np.mean(y_train) # 1 = inadiplência

# %%
np.mean(y_test) # 1 = inadiplência
# %%
classifier = LogisticRegression(C=0.1,
    class_weight=None,
    dual=False,
    fit_intercept=True,
    intercept_scaling=1,
    max_iter=100,
    multi_class='auto',
    n_jobs=None,
    penalty='l2',
    random_state=123,
    solver='liblinear',
    tol=0.0001,
    verbose=0,
    warm_start=False
)
# %%
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# %%
#! Determinar a acurácia
is_correct = y_pred == y_test
np.mean(is_correct)
# %%
classifier.score(X_test, y_test)
# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
# %%
from sklearn.dummy import DummyClassifier

dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit(X_train, y_train)
score_dummy = dummy_classifier.score(X_test, y_test)
print(f"Acurácia do dummy classifier é {score_dummy*100:.3f}%")
# %%
P = sum(y_test)
P #! amostras positivas
# %%
# verdadeiros positivos
TP = sum((y_test==1) & (y_pred==1))
TP
# %%
TPR = TP/P
TPR
# %%
# falsos negativos
FN = sum((y_test==1) & (y_pred==0))
FN
# %%
FNR = FN/P 
FNR
# %%
N = sum(y_test==0)
N
# %%
TN = sum((y_test==0) & (y_pred==0))
TN
# %%
FP = sum((y_test==0) & (y_pred==1))
FP
# %%
TNR = TN/N
FPR = FP/N 
print(f"The true positive rate is {TNR} and the false positive rate is {FPR}")
# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
# %%
from sklearn.metrics import plot_confusion_matrix
_ = plot_confusion_matrix(classifier, X_test, y_test)
# %%
