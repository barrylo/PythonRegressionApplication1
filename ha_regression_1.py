
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
#from sklearn import datasets, linear_model
from sklearn import linear_model
#import os
import pandas as pd

#-------- Orig -----
# Load the diabetes dataset
#diabetes = datasets.load_diabetes()
#-----------

#----------------

raw_data = {'target': [151.,   75.,  141.,  206.,  
                       135.,   97.,  138.,   63., 
                        110., 310.,  101.,   69.,  
                        179.,  185.,  118.,  171.,  
                        166.,  144., 97.,  168.,   
                        68.,   49.,   68.,  245.,  
                        184.,  202.,  137., 85.,  
                        131.,  283.,  129.,   59., 
                         341.,   87.,   65.,  102.,   
                         265.,  276.,  252.,   90., 
                          100.,   55.,   61.,   92.,
                            100.,   55.,   61.,   92.,
                              100.,   55.,   61.,   92.,
                                100.,   55.,   61.,   92.,
                          
                          ],
'data': [0.03807591,  0.05068012,  0.06169621,  -0.00259226,
         0.01990842, -0.01764613, -0.00188202, -0.04464164, 
         -0.05147406,  -0.03949338, -0.06832974, -0.09220405,
        0.08529891,  0.05068012,  0.04445121,  -0.00259226,
         0.00286377, -0.02593034, 0.04170844,  0.05068012, 
         -0.01590626,  -0.01107952, -0.04687948,  0.01549073,
       -0.04547248, -0.04464164,  0.03906215,  0.02655962,
         0.04452837, -0.02593034, -0.04547248, -0.04464164, 
         -0.0730303 ,  -0.03949338, -0.00421986,  0.00306441,
         -0.0730303 ,  -0.03949338, -0.00421986,  0.00306441,
         -0.0730303 ,  -0.03949338, -0.00421986,  0.00306441 ,
         -0.0730303 ,  -0.03949338, -0.00421986,  0.00306441 ,
         -0.0730303 ,  -0.03949338, -0.00421986,  0.00306441 ,
         -0.0730303 ,  -0.03949338, -0.00421986,  0.00306441  
          ]}


df = pd.DataFrame(raw_data, columns = ['target', 'data'])
df
df.to_csv('C:\PYDEV\gen_diabetes.csv')
df = pd.read_csv('C:\PYDEV\gen_diabetes.csv')
df
df = pd.read_csv('C:\PYDEV\gen_diabetes.csv')
#print(df)

diabetes_X_train = df.data[:-20]
diabetes_X_test = df.data[:-20]
#print(diabetes_X_test)
diabetes_y_train = df.target[:-20]
diabetes_y_test = df.target[:-20]
#print(diabetes_y_test)

#-----------------------

#-----------

# Split the data into training/testing sets
#diabetes_X_train = diabetes_X[:-20]
#diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
#diabetes_y_train = diabetes.target[:-20]
#diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
#regr.fit(diabetes_X_train, diabetes_y_train)
#regr.fit(diabetes_X_train.reshape(len(diabetes_X_train), 1), diabetes_y_train)

#print(diabetes_X_train.reshape(diabetes_X_train, 1))

diabetes_X_train = diabetes_X_train.reshape(int(len), 1)
diabetes_y_train = diabetes_y_train.reshape(int(len), 1)

diabetes_X_test = diabetes_X_test.reshape(int(len), 1)
diabetes_y_test = diabetes_y_test.reshape(int(len), 1)

#regr.fit(diabetes_X_train.reshape(len(diabetes_X_train), 1),
#         diabetes_y_train.reshape(len(diabetes_y_train), 1)
#          )
regr.fit(diabetes_X_train, diabetes_y_train)



# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()