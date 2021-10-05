# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 12:45:35 2021

@author: Hanuel
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')

# Things to do

# 1. Choose relevant columns
df.columns

df_model = df[['avg_salary','Rating', 'Size', 'Type of ownership', 'Industry', \
               'Sector', 'Revenue', 'num_comp', 'hourly', 'employer_provided', \
                   'job_state', 'same_state', 'age', 'python_yn', 'spark', 'aws', 'excel', \
                       'job_simp', 'seniority', 'desc_len']]

# 2. Get dummy data
df_dum = pd.get_dummies(df_model)


# 3. Create train-test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis = 1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. Make models (linear, lasso, rf)
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)

model.fit().summary()


# Sklearn linear regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))

# Lasso Regression

from sklearn.linear_model import Lasso
lm_l = Lasso()
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha = (i/100))
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3)))

plt.plot(alpha, error)

err = tuple(zip(alpha, error))

df_err = pd.DataFrame(err, columns = ['alpha', 'error'])

df_err[df_err.error == max(df_err.error)]

# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)

np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))


# 5. Tune models using GridSearchCV
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': range(10, 100, 10),
              'criterion': ['mse', 'mae'],
              'max_features': ['auto', 'sqrt', 'log2']}

gs = GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv=3)

gs.fit(X_train, y_train)

gs.best_score_

gs.best_estimator_


# 6. Test samples
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lml)
mean_absolute_error(y_test, tpred_rf)

mean_absolute_error(y_test, (tpred_lm + tpred_rf) / 2)

print(gs.best_estimator_)

import pickle
regressor = gs.best_estimator_
pickle.dump(regressor, open("model.pkl", "wb"))

filename = 'model.pkl'
model = pickle.load(open('model.pkl','rb'))

model.predict(X_test.iloc[1,:].values.reshape(1, -1))

list(X_test.iloc[1,:])
