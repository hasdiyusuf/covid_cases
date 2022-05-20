
import numpy as np



from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


#%%

class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def fillna(self,data):
        '''
        fill the nan values
        by using iterative 
        imputer

        Parameters
        ----------
        data : Dataframe
            DESCRIPTION.

        Returns
        -------
        data : Array
            DESCRIPTION.

        '''
        data = IterativeImputer().fit_transform(data)
        return data
    
    def features_splitting(self, data, window_size=30):
        '''
        splitting features data 

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        window_size : TYPE, optional
            DESCRIPTION. The default is 30.

        Returns
        -------
        lis : list
            DESCRIPTION.

        '''
        lis= []


        [lis.append(data[i-window_size:i,0]) for i in range(window_size,len(data))]
        lis = np.array(lis)
        
        return lis
            
    def target_splitting(self,data, window_size=30):
        '''
        split the target data

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        window_size : TYPE, optional
            DESCRIPTION. The default is 30.

        Returns
        -------
        lis : TYPE
            DESCRIPTION.

        '''
        lis = []
        
        [lis.append(data[i,0]) for i in range(window_size,len(data))]
        lis = np.array(lis)
        return lis

class ModelCreation():
    '''
    create lstm layer 
    
    '''
    def lstm_layer(self, model_path , data,  lstm_nodes = 64, dropout = (0.2)):
        model = Sequential()
        model.add(LSTM(lstm_nodes,activation ='tanh',
               return_sequences=(True),
               input_shape = (data.shape[1],1))) # return_sequences , LSTM only accept 3-dimension
        
        model.add(Dropout(dropout))
        model.add(LSTM(lstm_nodes))
        model.add(Dropout(dropout))
        model.add(Dense(1))

        model.summary()
        
        model.save(model_path)
        
        return model 
        
        
