#--------------------------------------------------------------------
#
# A base class is NeuralNet
# Derived classes are MLP, LSTMc, and ArbitraryNN
#
#--------------------------------------------------------------------

import metrics
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np

class NeuralNet(ABC):
    MAX_EPOCHS = 1000
    BATCH_SIZE = 128
    LOSS = 'mean_absolute_error'
    METRICS = [metrics.metric_IA, 'mean_squared_error']
    LEARNING_RATE = 1e-3
    ACTIVATION = 'sigmoid'
    NNEURONS = [8]

    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', NeuralNet.BATCH_SIZE)
        self.loss = kwargs.get('loss', NeuralNet.LOSS)
        self.metrics = kwargs.get('metrics', NeuralNet.METRICS)
        self.max_epochs = kwargs.get('metrics', NeuralNet.MAX_EPOCHS)
        self.learning_rate = kwargs.get('learning_rate', NeuralNet.LEARNING_RATE)
        self.activation = kwargs.get('activation', NeuralNet.ACTIVATION)
        self.nneurons = kwargs.get('nneurons', NeuralNet.NNEURONS[:])

        self.model = Sequential()
        #self.random_weights = None # used to re-initialize the model in new runs


    @abstractmethod
    def build_model(self):
        pass

    def train_model_with_valid(self, data_train_x, data_train_y, data_val_x, data_val_y):
        '''
        A training mode to define an optimal number of epochs using validation data.
        Returns a tuple (opt_epochs, opt_val_loss)
        '''

        #self.model.set_weights(self.random_weights) # re-initialize the model
        
        opt = Adam(learning_rate=self.learning_rate)
        
        self.model.compile(
            loss = self.loss,
            optimizer = opt,
            metrics =  self.metrics
        )
        
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100, verbose=0, mode='min', restore_best_weights=True)
        
        self.model.fit(
            data_train_x, data_train_y,
            verbose=0,
            shuffle=True,
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            validation_data=(data_val_x, data_val_y),
            callbacks=[earlystopping]
        )
        
        hist = self.model.history.history['val_loss']
        n_epochs_best = np.argmin(hist)
        scores_validation = self.model.evaluate(data_val_x, data_val_y, verbose=0, batch_size=data_val_x.shape[0])
        
        return (n_epochs_best, scores_validation[0])
    
    def train_model(self, data_x, data_y, opt_epochs):
        '''
        A training mode without validation when the optimal number of epochs is known.
        '''
        #self.model.set_weights(self.random_weights) # re-initialize the model
        
        opt = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            loss = self.loss,
            optimizer = opt,
            metrics =  self.metrics
        )
        self.model.fit(
            data_x, data_y,
            verbose=0,
            shuffle=True,
            batch_size=self.batch_size,
            epochs=int(opt_epochs),
            validation_split = 0.0
        )

    def predict(self, data_x):
        return self.model.predict(data_x, batch_size = data_x.shape[0])
            

class MLP(NeuralNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_model(self, input_shape):
        self.model = Sequential()
        self.model.add(Flatten(input_shape = input_shape))
        for nn in self.nneurons:
            self.model.add(Dense(nn, activation = self.activation))
        self.model.add(Dense(1, activation = 'linear'))

        #self.random_weights = self.model.get_weights() # save random weights to re-initialize the model in new runs

        
class LSTMc(NeuralNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_model(self, input_shape):
        self.model = Sequential()
        if len(self.nneurons) == 1:
            self.model.add(LSTM(self.nneurons[0], input_shape = input_shape, activation=self.activation))
        else:
            self.model.add(LSTM(self.nneurons[0], input_shape = input_shape, activation=self.activation, return_sequences=True))
            for nn in self.nneurons[1:-1]:
                self.model.add(LSTM(nn, activation=self.activation, return_sequences=True))
            self.model.add(LSTM(self.nneurons[-1], activation=self.activation))
            
        self.model.add(Dense(self.nneurons[-1], activation = self.activation))
        self.model.add(Dense(1, activation = 'linear'))

        #self.random_weights = self.model.get_weights() # save random weights to re-initialize the model in new runs

class ArbitraryNN(NeuralNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_model(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(16, input_shape = input_shape, activation = 'tanh', return_sequences=True))
        self.model.add(Dropout(0.1))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(16, activation = 'tanh'))
        self.model.add(Dropout(0.1))
        self.model.add(BatchNormalization())
        self.model.add(Dense(8, activation = 'sigmoid'))
        self.model.add(Dense(1, activation = 'linear'))

        #self.random_weights = self.model.get_weights() # save random weights to re-initialize the model in new runs







