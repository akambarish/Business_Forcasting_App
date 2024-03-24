# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn import  linear_model, metrics
from sklearn.model_selection import train_test_split



data = pd.read_csv('D:\Projects\Regression_Analysis\V2\Input_Data_File_Final_V2.csv')


y = data[['EPI']]
x = data[['Monthly_Spends_on_Performace_Marketing', 'no_of_Agents', 'LCR', 'Business_Last_year_Same_Month',
          'Website_leads','SEM_Leads','Other_Leads']]


reg = linear_model.LinearRegression()

reg.fit(x, y)


y_pred = reg.predict(x)

        
# pickling the model
import pickle
pickle_out = open("Prediction_model_v2.pkl", "wb")
pickle.dump(reg, pickle_out)
pickle_out.close()


