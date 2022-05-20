
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error 
from tensorflow.keras.callbacks import TensorBoard

from covid_cases_class import ExploratoryDataAnalysis, ModelCreation



#%% load the data
TRAIN_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
TEST_PATH =os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
LOG_PATH = os.path.join(os.getcwd(),'log')
MODEL_PATH =os.path.join(os.getcwd(),'model_saved','model.h5')

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

df_train['cases_new'] = pd.to_numeric(df_train['cases_new'], errors = 'coerce')
df_test['cases_new'] = pd.to_numeric(df_test['cases_new'], errors = 'coerce')

df_train.info()
df_train.describe().T
df_train.isna().sum()
df_test.isna().sum()



# drop object columns
df_train = df_train.drop(['date'],axis = 1)
df_test = df_test.drop(['date'],axis = 1)

#%% eda

eda = ExploratoryDataAnalysis() 

df_train_cleaned = eda.fillna(df_train)
df_test_cleaned = eda.fillna(df_test)

df_train = df_train_cleaned[:,0]
df_test = df_test_cleaned[:,0]

# visualize the train data
plt.figure()
plt.plot(df_train)
plt.show()



# Data Preprocessing
mms = MinMaxScaler()
df_train_scaled = mms.fit_transform(np.expand_dims(df_train, axis = -1))
df_test_scaled = mms.transform(np.expand_dims(df_test, axis = -1))

#%% data splitting

x_train = eda.features_splitting(df_train_scaled, window_size=30)
y_train = eda.target_splitting(df_train_scaled, window_size=30)

# concate test data with train data 
data = np.concatenate((df_train_scaled, df_test_scaled), axis = 0)
data_total = data[-130:] # 30 days + 100 (no. of datasets)

x_test = eda.features_splitting(data_total, window_size=30)
y_test = eda.target_splitting(data_total, window_size=30)

# expand the dimensions 
x_train = np.expand_dims(x_train, axis = -1)
x_test = np.expand_dims(x_test, axis = -1)

#%% model creation

mc = ModelCreation()

model = mc.lstm_layer(MODEL_PATH, x_train)

#%% model compile/ model fitting

log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# tensorboard callbacks
tensorboard_callbacks = TensorBoard(log_dir = log_dir)




model.compile(optimizer = 'adam',
              loss = 'mse',
              metrics = 'mse')

model.fit(x_train, y_train, epochs = 100 ,batch_size=64, callbacks = [tensorboard_callbacks])


#%% evaluate model

predicted = []
for test in x_test:
    predicted.append(model.predict(np.expand_dims(test, axis = 0)))

predicted = np.array(predicted)

#%% model analysis
plt.figure()
plt.plot(predicted.reshape(len(predicted),1), color = 'r', label = 'predicted values')
plt.plot(y_test, color = 'b', label = 'actual values')
plt.legend(['predicted', 'actual'])
plt.show()

#%% model evaluation

y_true = y_test
y_pred = predicted.reshape(len(predicted),1)

print(mean_absolute_error(y_test,y_pred)/sum(abs(y_true))*100)


# The MAPE for this model is 1.4% which
# does not able to achieve lesser than 1%

# The recommendation to reduce number of error probably by using
# machine learnign approach or to add more nodes in the LSTM layer



 