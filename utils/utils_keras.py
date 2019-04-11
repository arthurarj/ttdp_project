# Keras utils
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

# Metrics
def mae(y_true, y_pred):
    return K.mean(K.abs(y_true-y_pred))
def mape(y_true, y_pred):
    return 100*K.mean(K.abs((y_true-y_pred)/y_true))
def mare(y_true, y_pred):
    return 100*K.sum(K.abs(y_true-y_pred))/K.sum(K.abs(y_true))

# Losses
def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

def plot_keras_graphs(model):
    plt.figure(figsize=(30,4))
    plt.subplot(1,4,1)
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('LOSS')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')

    plt.subplot(1,4,2)
    plt.plot(model.history.history['mae'])
    plt.plot(model.history.history['val_mae'])
    plt.title('MAE')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')

    plt.subplot(1,4,3)
    plt.plot(model.history.history['mape'])
    plt.plot(model.history.history['val_mape'])
    plt.title('MAPE')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')

    plt.subplot(1,4,4)
    plt.plot(model.history.history['mare'])
    plt.plot(model.history.history['val_mare'])
    plt.title('MARE')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')

    plt.show()