from sklearn.linear_model import Ridge
import numpy as np

from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import RandomizedSearchCV


import PyPluMA
import PyIO
class RandomizedSearchCVPlugin:
    def input(self, inputfile):
        self.parameters = PyIO.readParameters(inputfile)
        self.data_path = PyPluMA.prefix()+"/"+self.parameters["csvfile"]
        self.columns_rfe = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["features"]
)
    def run(self):
        pass

    def output(self, outputfile):
       data_df = pd.read_csv(self.data_path)

       # # Tramsform categorical data to categorical format:
       # for category in categorical_cols:
       #     data_df[category] = data_df[category].astype('category')
       #

       # Clean numbers:
       #"Cocain_Use": {"yes":1, "no":0},
       cleanup_nums = { "Cocain_Use": {"yes":1, "no":0},
                 "race": {"White":1, "Black":0, "BlackIsraelite":0, "Latina":1},
       }

       data_df.replace(cleanup_nums, inplace=True)

       # Drop id column:
       data_df = data_df.drop(["pilotpid"], axis=1)

       # remove NaN:
       data_df = data_df.fillna(0)

       # Standartize variables
       from sklearn import preprocessing
       names = data_df.columns
       scaler = preprocessing.StandardScaler()
       data_df_scaled = scaler.fit_transform(data_df)
       data_df_scaled = pd.DataFrame(data_df_scaled, columns=names)


       y_col = "interleukin6"
       test_size = 0.25
       validate = True
       random_state = 2

       y = data_df[y_col]

       X = data_df_scaled.drop([y_col], axis=1)

       y_col = "interleukin6"
       validate = True


       X_rfe = X[self.columns_rfe]


       X_train_rfe, X_valid_rfe, y_train_rfe, y_valid_rfe = train_test_split(X_rfe, y, test_size = test_size, random_state = random_state)



       ridge_params = {'alpha':[10,50,100,200, 230, 250,265, 270, 275, 290, 300, 500]}

       ln = Ridge()

       ln_random = RandomizedSearchCV(estimator = ln, param_distributions = ridge_params, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
       # Fit the random search model
       ln_random.fit(X_train_rfe, y_train_rfe)

       ln_best = ln_random.best_estimator_
       ln_best.fit(X_train_rfe, y_train_rfe)
       print(ln_best)
       print('Training R^2: {:.2f} \nTest R^2: {:.2f}'.format(ln_best.score(X_train_rfe, y_train_rfe), ln_best.score(X_valid_rfe, y_valid_rfe)))

       scores = cross_val_score(ln_best, X, y, cv=5)
       print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


