import keras
from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

from constants2 import *





def create_cnn(X, y, model_f_name):
    
    
    model = Sequential()

    model.add(
        Convolution1D(                    
            6, 
            51,
            input_dim = X.shape[2], #3
            init='uniform', 
            border_mode='same')) 

    model.add(Activation('relu'))

    model.add(
        Convolution1D(
            1, # output_dim,                    
            X.shape[1],
            init='uniform',
            border_mode='valid'))

    model.add(Activation('relu'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy', 'mean_squared_error'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(MODELS_PATH + model_f_name, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.fit(X_train, y_train,
              batch_size=100,
              epochs=30,
              verbose=1, 
              callbacks = [es, mc], 
              validation_data=(X_test, y_test))
    
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X)

    
    r2_train = r2_score(y_train.reshape(1,-1), y_pred_train.reshape(1,-1))
    r2_test = r2_score(y_test.reshape(1,-1), y_pred_test.reshape(1,-1))
    print()
    print('R^2 training: %s,\nR^2 test: %s'%(r2_train, r2_test))
    
    
    try:
        snr = np.quantile(list(filter(lambda x: x >= 1, y_pred)), 0.9)/np.quantile(list(filter(lambda x: x >= 1, y_pred)), 0.1)
        print('SNR =', snr)
    except:
        pass
    
    print(MODELS_PATH + model_f_name)
    return model

   