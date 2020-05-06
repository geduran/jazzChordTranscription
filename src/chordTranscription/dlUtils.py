import numpy                as np
import keras
from   keras.models         import Model
from   keras.layers         import Dense, Conv2D, Flatten, Activation, Dropout
from   keras.layers         import regularizers, MaxPooling2D, Lambda
from   keras.layers         import GRU, LSTM, SimpleRNN, BatchNormalization
from   keras.layers         import Bidirectional, TimeDistributed, Input
from   keras.callbacks      import ModelCheckpoint, EarlyStopping, Callback
from   keras.utils.np_utils import to_categorical
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
import sklearn.model_selection
import random
import pickle
import glob
import os



# class My_Generator(Sequence):
#     """
#         Class to manage samples at training. Dataset is too big to load it in
#         RAM, so this class manages to load one batch at a time.
#     """
#
#     def __init__(self, image_filenames, labels, batch_size, xSize, ySize):
#         self.xSize = xSize
#         self.ySize = ySize
#         self.image_filenames, self.labels = image_filenames, labels
#         self.batch_size = batch_size
#
#     def __len__(self):
#         return np.ceil(len(self.image_filenames) / float(self.batch_size)).astype(int)
#
#     def __getitem__(self, idx):
#
#         ############################## TODO ######################################
#
#         batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
#         y_out = np.array(batch_y)
#         x_out = np.zeros((len(batch_x), 3, self.xSize, self.ySize))
#         for file_name, i in zip(batch_x, range(len(batch_x))):
#             im = imread(file_name)/255.
#             x_out[i, 0, :, :] = im[:,:,0]
#             x_out[i, 1, :, :] = im[:,:,1]
#             x_out[i, 2, :, :] = im[:,:,2]
#         return x_out, y_out



# my_training_batch_generator = My_Generator(training_filenames, GT_training, batch_size, roiSize, roiSize)
# my_validation_batch_generator = My_Generator(validation_filenames, GT_validation, batch_size, roiSize, roiSize)
#
#
# history = model.fit_generator(generator=my_training_batch_generator,
#                                       steps_per_epoch = (num_training_samples // batch_size),
#                                       epochs          = epochs,
#                                       verbose         = 1,
#                                       validation_data = my_validation_batch_generator,
#                                       validation_steps= (num_validation_samples // batch_size),
#                                       use_multiprocessing=True,
#                                       workers         =16,
#                                       max_queue_size  =32,
#                                       # class_weight    = {0: 1., 1: class_proportion},
#                                       shuffle         = True,
#                                       callbacks       = callbacks
#                                       )
#




def functional_encoder(input_shape):
    print('En functional_encoder input_shape: {}'.format(input_shape))

    input = Input(shape=input_shape)

    conv1 = Conv2D(4,  kernel_size=(5, 5), padding='same',
                   activation='relu', data_format='channels_last')(input)

    conv1_bn = BatchNormalization()(conv1)

    conv2 = Conv2D(4,  kernel_size=(3, 3), padding='same',
                   activation='relu', data_format='channels_last')(conv1_bn)

    conv2_bn = BatchNormalization()(conv2)

    conv3 = Conv2D(1,  kernel_size=1, padding='same',
                   activation='relu', data_format='channels_last')(conv2_bn)

    conv3_bn = BatchNormalization()(conv3)

    conv4 = Conv2D(24, (1, int(conv1.shape[2])), padding='valid',
                   activation='relu', data_format='channels_last')(conv3_bn)

    conv4_bn = BatchNormalization()(conv4)

    sq_conv4 = Lambda(lambda x: K.squeeze(x, axis=2))(conv4_bn)

    gru1 = Bidirectional(GRU(64, return_sequences=True))(sq_conv4)

    root_output = TimeDistributed(Dense(13, activation='softmax'), name='root')(gru1)

    notes_output = TimeDistributed(Dense(13, activation='sigmoid'), name='notes')(gru1)

    gru2 = Bidirectional(GRU(64, return_sequences=True))(sq_conv4)

    beat_output = TimeDistributed(Dense(2, activation='softmax'), name='beats')(gru2)

    model = Model(inputs=input, outputs=[root_output, notes_output, beat_output])


    losses = {
    	'root': 'categorical_crossentropy',
    	'notes': 'mean_squared_error',
        'beats': 'binary_crossentropy'
    }


    model.compile(optimizer='adam',
                  loss= losses,
                  metrics=['accuracy'])

    model.summary()
    return model



def define_callBacks(model_file):
    #metrics = Metrics()
    callbacks = [
        EarlyStopping(
            monitor        = 'val_acc',
            patience       = 10,
            mode           = 'max',
            verbose        = 1),
        ModelCheckpoint(model_file,
            monitor        = 'val_acc',
            save_best_only = True,
            mode           = 'max',
            verbose        = 0)#,
        #metrics
    ]
    return callbacks



    data = {
            'data': warped_cqts,
            'averaged_data': averaged_cqt,
            'root_labels': coded_roots,
            'notes_labels': coded_notes,
            'beat_labels': beat_labels,
            'hop_len': hop_len,
            'sr': sr,
            'method': method
            }



def load_data(path, averaged=False, seq_len=100):
    root_labels = None
    notes_labels = None
    beats_labels = None
    all_data = None
    if averaged:
        data_field = 'averaged_data'
    else:
        data_field = 'data'

    # shape is features x len_seq
    all_files = glob.glob(path+'*pkl')

    for curr_file in all_files:
        data = _open_pickle(curr_file)
        curr_root_labels = data['root_labels']
        curr_notes_labels = data['notes_labels']
        curr_beats_labels = to_categorical(data['beats_labels']).T
        curr_data = data[data_field]

        if isinstance(curr_data, list):
            for curr_seg in curr_data:
                root_labels = _update_array(root_labels, curr_root_labels)
                notes_labels = _update_array(notes_labels, curr_notes_labels)
                beats_labels = _update_array(beats_labels, curr_beats_labels)
                all_data = _update_array(all_data, curr_seg)


    print('Loaded: ')
    print('  root_labels: {}'.format(root_labels.shape))
    print('  notes_labels: {}'.format(notes_labels.shape))
    print('  beats_labels: {}'.format(beats_labels.shape))
    print('  all_data: {}'.format(all_data.shape))


    ind = np.arange(0, all_data.shape[1] - seq_len, int(seq_len/8))
    n_samples = len(ind)

    n_features = all_data.shape[0]
    n_channels = 1

    shaped_data = np.zeros((n_samples, seq_len, n_features))
    shaped_root_labels = np.zeros((n_samples, seq_len, root_labels.shape[0]))
    shaped_notes_labels = np.zeros((n_samples, seq_len, notes_labels.shape[0]))
    shaped_beats_labels = np.zeros((n_samples, seq_len, beats_labels.shape[0]))


    cont = 0
    for i in ind:
        print('Cargando Datos {}/{}'.format(cont,n_samples), end='\r')
        shaped_data[cont,:,:] = all_data[:,i:i+seq_len].T
        shaped_root_labels[cont,:,:] = root_labels[:,i:i+seq_len].T
        shaped_notes_labels[cont,:,:] = notes_labels[:,i:i+seq_len].T
        shaped_beats_labels[cont,:,:] = beats_labels[:,i:i+seq_len].T
        cont += 1

    split_data = sklearn.model_selection.train_test_split(shaped_data, shaped_root_labels,
                                                          shaped_notes_labels, shaped_beats_labels,
                                                          test_size=0.05)

    train_d, test_d, train_r, test_r, train_n, test_n, train_b, test_b = split_data

    print('train_d: {}, test_d: {}'.format(train_d.shape, test_d.shape))
    print('train_r: {}, test_r: {}'.format(train_r.shape, test_r.shape))
    print('train_n: {}, test_n: {}'.format(train_n.shape, test_n.shape))
    print('train_b: {}, test_b: {}'.format(train_b.shape, test_b.shape))
    return (train_d, test_d, train_r, test_r, train_n, test_n, train_b, test_b)

def _open_pickle(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def _update_array(array, new_values, axis=1):
    if array is None:
        return new_values
    else:
        array = np.concatenate((array, new_values), axis=axis)
        return array




#
# input_shape = (None, 84, 1)
# functional_encoder(input_shape)
#
