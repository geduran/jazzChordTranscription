import numpy                as np
import keras
from   keras.models         import Model, Sequential
from   keras.layers         import Dense, Conv2D, Flatten, Activation, Dropout
from   keras.layers         import regularizers, MaxPooling2D, Lambda, Concatenate
from   keras.layers         import GRU, LSTM, SimpleRNN, BatchNormalization
from   keras.layers         import Bidirectional, TimeDistributed, Input
from   keras.callbacks      import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from   keras.utils.np_utils import to_categorical
import keras.backend        as K
import tensorflow           as tf
from   sklearn.metrics      import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn              import preprocessing
from   collections          import deque
import pandas               as     pd
import random
import pickle
import glob
import sys
import os



class EncoderDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, seq_len=100, num_features=84, n_channels=1,
                 shuffle=True, labels_r_classes=13, labels_n_classes=13, labels_b_classes=2):
        'Initialization'
        self.seq_len = seq_len
        self.num_features = num_features
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_labels_r = labels_r_classes
        self.n_labels_n = labels_n_classes
        self.n_labels_b = labels_b_classes
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, yr, yn, yb = self.__data_generation(list_IDs_temp)
        labels = {'root': yr, 'notes': yn, 'beats': yb}

        return X, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.seq_len, self.num_features))
        yr = np.empty((self.batch_size, self.seq_len, self.n_labels_r), dtype=int)
        yn = np.empty((self.batch_size, self.seq_len, self.n_labels_n), dtype=int)
        yb = np.empty((self.batch_size, self.seq_len, self.n_labels_b), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            with open(ID, 'rb') as f:
                c_d, c_r, c_n, c_b = pickle.load(f)
            # Store sample
            X[i,]  = c_d
            yr[i,] = c_r
            yn[i,] = c_n
            yb[i,] = c_b

        X = X.reshape((*X.shape, 1))
        return X, yr, yn, yb


class BothDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, seq_len=100, num_features=84, n_channels=1,
                 shuffle=True, labels_r_classes=13, labels_n_classes=12, labels_b_classes=2,
                 labels_chord_classes=61):
        'Initialization'
        self.seq_len = seq_len
        self.num_features = num_features
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_labels_r = labels_r_classes
        self.n_labels_n = labels_n_classes
        self.n_labels_b = labels_b_classes
        self.n_labels_chord = labels_chord_classes
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, yr, yn, yb, Y = self.__data_generation(list_IDs_temp)
        labels = {'root': yr, 'notes': yn, 'beats': yb, 'chords': Y}

        return X, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.seq_len, self.num_features))
        yr = np.empty((self.batch_size, self.seq_len, self.n_labels_r), dtype=int)
        yn = np.empty((self.batch_size, self.seq_len, self.n_labels_n), dtype=int)
        yb = np.empty((self.batch_size, self.seq_len, self.n_labels_b), dtype=int)
        Y = np.empty((self.batch_size, self.seq_len, self.n_labels_chord), dtype=int)


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            with open(ID, 'rb') as f:
                c_d, c_r, c_n, c_b, y = pickle.load(f)
            # Store sample
            X[i,]  = c_d
            yr[i,] = c_r
            yn[i,] = c_n
            yb[i,] = c_b
            Y[i,]  = y

        X = X.reshape((*X.shape, 1))
        return X, yr, yn, yb, Y



class DecoderDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=1, seq_len=None, num_features=26, n_channels=1,
                 shuffle=True, n_labels=61):
        'Initialization'
        self.seq_len = seq_len
        self.num_features = num_features
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_labels = n_labels
        self.shuffle = shuffle
        self._n_samples = len(list_IDs)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Current index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, labels = self.__data_generation(list_IDs_temp)

        return X, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        X = np.empty((self.batch_size, self.seq_len, self.num_features))
        Y = np.empty((self.batch_size, self.seq_len, self.n_labels), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
        # Generate data
            with open(ID, 'rb') as f:
                x, y = pickle.load(f)
            X[i,]  = x
            Y[i,] = y
        # X = X.reshape((*X.shape, 1))
        return X, Y



#
# class DecoderDataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, batch_size=1, seq_len=None, num_features=26, n_channels=1,
#                  shuffle=True, n_labels=61):
#         'Initialization'
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.batch_size = batch_size
#         self.list_IDs = list_IDs
#         self.n_labels = n_labels
#         self.shuffle = shuffle
#         self._n_samples = len(list_IDs)
#         self.on_epoch_end()
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return self._n_samples
#
#     def __getitem__(self, index):
#         'Generate one batch of data'
#
#         # Current index
#         curr_index = self.indexes[index]
#
#         # Generate data
#         X, labels = self.__data_generation(self.list_IDs[curr_index])
#
#         return X, labels
#
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, ID_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#
#         # Generate data
#         with open(ID_temp, 'rb') as f:
#             X, Y = pickle.load(f)
#
#         # X = X.reshape((*X.shape, 1))
#         return X, Y


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.5):
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_pred[:,1], 0, 1)) * y_true[:,1])
        possible_positives = K.sum(y_true[:,1])
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_pred[:,1], 0, 1)) * y_true[:,1])
        predicted_positives = K.sum(K.round(K.clip(y_pred[:,1], 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def functional_both(input_shape):

    input = Input(shape=input_shape)

    input_bn = BatchNormalization()(input)

    conv1 = Conv2D(1,  kernel_size=(5, 5), padding='same',
                   activation='relu', data_format='channels_last')(input_bn)

    conv1_bn = BatchNormalization()(conv1)
    #
    # conv2 = Conv2D(4,  kernel_size=(3, 3), padding='same',
    #                activation='relu', data_format='channels_last')(conv1_bn)
    #
    # conv2_bn = BatchNormalization()(conv2)
    #
    # conv3 = Conv2D(4,  kernel_size=(3, 3), padding='same',
    #                activation='relu', data_format='channels_last')(conv2_bn)
    #
    # conv3_bn = BatchNormalization()(conv3)

    # For root and notes
    conv4 = Conv2D(84, kernel_size=(1, int(conv1.shape[2])), padding='valid',
                   activation='relu', data_format='channels_last')(conv1_bn)

    conv4_bn = BatchNormalization()(conv4)


    # conv4_bn = Conv2D(64,  kernel_size=(1, 1), padding='same',
    #                activation='relu', data_format='channels_last')(conv4_bn)
    #
    # conv4_bn = BatchNormalization()(conv4_bn)


    sq_conv4 = Lambda(lambda x: K.squeeze(x, axis=2), name='sq_conv4')(conv4_bn)

    gru1_interm = Bidirectional(GRU(128, dropout=0.35, recurrent_dropout=0.35,
                         # activity_regularizer=regularizers.l2(1e-3),
                         return_sequences=True))(sq_conv4)

    gru1_interm_bn = BatchNormalization()(gru1_interm)

    gru2_interm = Bidirectional(GRU(128, dropout=0.35, recurrent_dropout=0.35,
                         # activity_regularizer=regularizers.l2(1e-3),
                         return_sequences=True))(gru1_interm_bn)

    gru2_interm_bn = BatchNormalization()(gru2_interm)

    root_output = TimeDistributed(Dense(13, activation='softmax'), name='root')(gru2_interm_bn)

    notes_output = TimeDistributed(Dense(12, activation='sigmoid'), name='notes')(gru2_interm_bn)

    #
    # # For beats
    # conv5 = Conv2D(4,  kernel_size=(3, 3), padding='same',
    #                activation='relu', data_format='channels_last')(conv1_bn)
    #
    # conv5_bn = BatchNormalization()(conv5)
    #
    # conv6 = Conv2D(1,  kernel_size=(3, 3), padding='same',
    #                activation='relu', data_format='channels_last')(conv5_bn)
    #
    # conv6_bn = BatchNormalization()(conv6)
    #
    # conv6_mp = MaxPooling2D(pool_size=(1,2), data_format='channels_last')(conv6_bn)
    #
    # sq_conv6 = Lambda(lambda x: K.squeeze(x, axis=3))(conv6_mp)
    #
    # gru1_beat = Bidirectional(GRU(32, return_sequences=True, dropout=0.35,
    #                      recurrent_dropout=0.35))(sq_conv6)
    #
    # gru1_beat_bn = BatchNormalization()(gru1_beat)
    #
    gru2_beat = Bidirectional(GRU(32, return_sequences=True, dropout=0.3,
                         recurrent_dropout=0.3))(sq_conv4)

    gru2_beat_bn = BatchNormalization()(gru2_beat)

    beat_output = TimeDistributed(Dense(2, activation='softmax'), name='beats')(gru2_beat_bn)



    concat = Concatenate(axis=2, name='concat')([sq_conv4, root_output, notes_output])#,
                                  # beat_output])

    concat_bn = BatchNormalization(name='concat_bn')(concat)

    gru1_chord = Bidirectional(GRU(128, return_sequences=True, dropout=0.35,
                         recurrent_dropout=0.35), name='gru1_chord')(concat_bn)

    gru1_chord_bn = BatchNormalization(name='gru1_chord_bn')(gru1_chord)

    gru2_chord = Bidirectional(GRU(128, return_sequences=True, dropout=0.35,
                         recurrent_dropout=0.35), name='gru2_chord')(gru1_chord_bn)

    gru2_chord_bn = BatchNormalization(name='gru2_chord_bn')(gru2_chord)

    chord_output = TimeDistributed(Dense(61, activation='softmax'), name='chords')(gru2_chord_bn)


    model = Model(inputs=input, outputs=[root_output, notes_output, beat_output, chord_output])


    losses = {
    	'root': 'categorical_crossentropy',
    	'notes': 'mean_squared_error',
        'beats': 'binary_crossentropy',
        'chords': 'categorical_crossentropy'
    }
    losses_weight = {
        	'root': 1,
        	'notes': 10,
            'beats': 5,
            'chords': 3
    }

    metrics = {
        'root': 'accuracy',
        'notes': 'accuracy',
        'beats': [f1],
        'chords': 'accuracy'
    }

    model.compile(optimizer='adam',
                  loss= losses,
                  loss_weights=losses_weight,
                  metrics=metrics)

    # model.summary()
    return model

def only_decoder(input_shape):

    input = Input(shape=input_shape)

    concat_bn = BatchNormalization(name='concat_bn')(input)

    gru1_chord = Bidirectional(GRU(128, return_sequences=True, dropout=0.2,
                         recurrent_dropout=0.2), name='gru1_chord')(concat_bn)

    gru1_chord_bn = BatchNormalization(name='gru1_chord_bn')(gru1_chord)

    gru2_chord = Bidirectional(GRU(128, return_sequences=True, dropout=0.2,
                         recurrent_dropout=0.2), name='gru2_chord')(gru1_chord_bn)

    gru2_chord_bn = BatchNormalization(name='gru2_chord_bn')(gru2_chord)

    chord_output = TimeDistributed(Dense(61, activation='softmax'), name='chords')(gru2_chord_bn)

    model = Model(inputs=input, outputs=[chord_output])

    model.compile(optimizer='adam',
                  loss= 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def encoder_callBacks(model_file):
    #metrics = Metrics()
    callbacks = [
        EarlyStopping(
            monitor        = 'root_acc',
            patience       = 10,
            mode           = 'max',
            verbose        = 1),
        ModelCheckpoint(model_file,
            monitor        = 'root_acc',
            save_best_only = False,
            mode           = 'max',
            verbose        = 0),
        ReduceLROnPlateau(
            patience=5,
            factor=0.2,
            verbose=1,
            monitor='chords_acc')

    ]
    return callbacks

def decoder_callBacks(model_file):
    #metrics = Metrics()
    callbacks = [
        EarlyStopping(
            monitor        = 'val_acc',
            patience       = 10,
            mode           = 'max',
            verbose        = 1),
        ModelCheckpoint(model_file,
            monitor        = 'val_acc',
            save_best_only = False,
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



def encoder_split_data(path, averaged=False, seq_len=100, out_dir='./temp_data_encoder/'):


    root_labels = None
    notes_labels = None
    beats_labels = None
    all_data = None

    val_root_labels = None
    val_notes_labels = None
    val_beats_labels = None
    val_all_data = None

    if averaged:
        data_field = 'averaged_data'
    else:
        data_field = 'data'

    # shape is features x len_seq
    all_songs = glob.glob(path+'*pkl')

    validation_songs = random.sample(all_songs, 3)
    ii=0
    for curr_song in all_songs:
        print('Opening song {}/{}              '.format(ii, len(all_songs)), end='\r')
        ii += 1
        data = _open_pickle(curr_song)
        curr_root_labels = data['root_labels']
        curr_notes_labels = data['notes_labels']
        curr_beats_labels = to_categorical(data['beats_labels']).T
        curr_data = data[data_field]
        if not isinstance(curr_data, list):
            curr_data = [curr_data]

        if curr_song in validation_songs:
            for curr_seg in curr_data:
                val_root_labels = _update_array(val_root_labels, curr_root_labels)
                val_notes_labels = _update_array(val_notes_labels, curr_notes_labels)
                val_beats_labels = _update_array(val_beats_labels, curr_beats_labels)
                val_all_data = _update_array(val_all_data, curr_seg)
        else:
            for curr_seg in curr_data:
                root_labels = _update_array(root_labels, curr_root_labels)
                notes_labels = _update_array(notes_labels, curr_notes_labels)
                beats_labels = _update_array(beats_labels, curr_beats_labels)
                all_data = _update_array(all_data, curr_seg)

    print('Loaded: ')
    print('  root_labels: {}, val: {}'.format(root_labels.shape, val_root_labels.shape))
    print('  notes_labels: {}, val {}'.format(notes_labels.shape, val_notes_labels.shape))
    print('  beats_labels: {}, val {}'.format(beats_labels.shape, val_beats_labels.shape))
    print('  all_data: {}, val {}'.format(all_data.shape, val_all_data.shape))

    mean = np.mean(np.concatenate((all_data, all_data), axis=1), axis=1)
    std = np.std(np.concatenate((all_data, all_data), axis=1), axis=1)

    all_data = ((all_data.T - mean) / std).T
    val_all_data = ((val_all_data.T - mean) / std).T

    ind = np.arange(0, all_data.shape[1] - seq_len, int(seq_len/5))
    n_samples = len(ind)
    val_ind = np.arange(0, val_all_data.shape[1] - seq_len, int(seq_len/5))
    val_n_samples = len(val_ind)

    n_features = all_data.shape[0]
    n_channels = 1

    shaped_data = np.zeros((n_samples, seq_len, n_features))
    shaped_root_labels = np.zeros((n_samples, seq_len, root_labels.shape[0]))
    shaped_notes_labels = np.zeros((n_samples, seq_len, notes_labels.shape[0]))
    shaped_beats_labels = np.zeros((n_samples, seq_len, beats_labels.shape[0]))

    val_shaped_data = np.zeros((val_n_samples, seq_len, n_features))
    val_shaped_root_labels = np.zeros((val_n_samples, seq_len, root_labels.shape[0]))
    val_shaped_notes_labels = np.zeros((val_n_samples, seq_len, notes_labels.shape[0]))
    val_shaped_beats_labels = np.zeros((val_n_samples, seq_len, beats_labels.shape[0]))

    cont = 0
    for i in ind:
        print('Cargando Datos {}/{}'.format(cont,n_samples), end='\r')
        shaped_data[cont,:,:] = all_data[:,i:i+seq_len].T
        shaped_root_labels[cont,:,:] = root_labels[:,i:i+seq_len].T
        shaped_notes_labels[cont,:,:] = notes_labels[:,i:i+seq_len].T
        shaped_beats_labels[cont,:,:] = beats_labels[:,i:i+seq_len].T
        cont += 1

    cont = 0
    for i in val_ind:
        print('Cargando Datos {}/{}'.format(cont,val_n_samples), end='\r')
        val_shaped_data[cont,:,:] = val_all_data[:,i:i+seq_len].T
        val_shaped_root_labels[cont,:,:] = val_root_labels[:,i:i+seq_len].T
        val_shaped_notes_labels[cont,:,:] = val_notes_labels[:,i:i+seq_len].T
        val_shaped_beats_labels[cont,:,:] = val_beats_labels[:,i:i+seq_len].T
        cont += 1


    print('train_d: {}, val_d: {}'.format(shaped_data.shape, val_shaped_data.shape))
    print('train_r: {}, val_r: {}'.format(shaped_root_labels.shape, val_shaped_root_labels.shape))
    print('train_n: {}, val_n: {}'.format(shaped_notes_labels.shape, val_shaped_notes_labels.shape))
    print('train_b: {}, val_b: {}'.format(shaped_beats_labels.shape, val_shaped_beats_labels.shape))

    n_train = shaped_data.shape[0]
    n_val = val_shaped_data.shape[0]

    _makeTempDir(out_dir)
    for i, data in enumerate(zip(shaped_data, shaped_root_labels, shaped_notes_labels, shaped_beats_labels)):
        t_d, t_r, t_n, t_b = data
        curr_name = 'train_sample_' + str(i) + '.pkl'
        curr_data = (t_d, t_r, t_n, t_b)
        _save_pkl(curr_data, out_dir + '/' + curr_name)

    for i, data in enumerate(zip(val_shaped_data, val_shaped_root_labels, val_shaped_notes_labels, val_shaped_beats_labels)):
        t_d, t_r, t_n, t_b = data
        curr_name = 'validation_sample_' + str(i) + '.pkl'
        curr_data = (t_d, t_r, t_n, t_b)
        _save_pkl(curr_data, out_dir + curr_name)

    return n_train, n_val



def both_split_data(path, seq_len=100, out_dir='./temp_data_encoder/'):

    CE = chordEval()
    root_labels = None
    notes_labels = None
    beats_labels = None
    chord_labels = None
    all_data = None

    val_root_labels = None
    val_notes_labels = None
    val_beats_labels = None
    val_chord_labels = None
    val_all_data = None


    # shape is features x len_seq
    all_files = glob.glob(path+'*pkl')
    song_names = set()

    for curr_file in all_files:
        song_names.add(curr_file.split('2048')[0])

    validation_songs = random.sample(song_names, 3)


    ii=0
    for curr_song in all_files:
        print('Opening song {}/{}              '.format(ii, len(all_files)), end='\r')
        ii += 1
        data = _open_pickle(curr_song)
        curr_root_labels = data['root_labels']
        curr_notes_labels = data['notes_labels']
        curr_beats_labels = to_categorical(data['beats_labels']).T

        name_root = CE.root_name(CE.root_predictions(curr_root_labels))
        name_chord, _ = CE.chord_type_greedy(CE.root_predictions(curr_root_labels), curr_notes_labels)

        curr_chord_labels = to_categorical(encode_labels(name_root,
                                           name_chord),num_classes=61).T

        curr_data = data['data']

        if not isinstance(curr_data, list):
            curr_data = [curr_data]

        if curr_song.split('2048')[0] in validation_songs:
            for curr_seg in curr_data:
                val_root_labels = _update_array(val_root_labels, curr_root_labels)
                val_notes_labels = _update_array(val_notes_labels, curr_notes_labels)
                val_beats_labels = _update_array(val_beats_labels, curr_beats_labels)
                val_chord_labels = _update_array(val_chord_labels, curr_chord_labels)
                val_all_data = _update_array(val_all_data, curr_seg)
        else:
            for curr_seg in curr_data:
                root_labels = _update_array(root_labels, curr_root_labels)
                notes_labels = _update_array(notes_labels, curr_notes_labels)
                beats_labels = _update_array(beats_labels, curr_beats_labels)
                chord_labels = _update_array(chord_labels, curr_chord_labels)
                all_data = _update_array(all_data, curr_seg)

    print('Loaded: ')
    print('  root_labels: {}, val: {}'.format(root_labels.shape, val_root_labels.shape))
    print('  notes_labels: {}, val {}'.format(notes_labels.shape, val_notes_labels.shape))
    print('  beats_labels: {}, val {}'.format(beats_labels.shape, val_beats_labels.shape))
    print('  chord_labels: {}, val {}'.format(chord_labels.shape, val_chord_labels.shape))
    print('  all_data: {}, val {}'.format(all_data.shape, val_all_data.shape))

    # mean = np.mean(np.concatenate((all_data, val_all_data), axis=1), axis=1)
    # std = np.std(np.concatenate((all_data, val_all_data), axis=1), axis=1)
    # all_data = ((all_data.T - mean) / std).T
    # val_all_data = ((val_all_data.T - mean) / std).T


    ind = np.arange(0, all_data.shape[1] - seq_len, int(seq_len/4))
    n_samples = len(ind)
    val_ind = np.arange(0, val_all_data.shape[1] - seq_len, int(seq_len/4))
    val_n_samples = len(val_ind)

    n_features = all_data.shape[0]
    n_channels = 1

    shaped_data = np.zeros((n_samples, seq_len, n_features))
    shaped_root_labels = np.zeros((n_samples, seq_len, root_labels.shape[0]))
    shaped_notes_labels = np.zeros((n_samples, seq_len, notes_labels.shape[0]))
    shaped_beats_labels = np.zeros((n_samples, seq_len, beats_labels.shape[0]))
    shaped_chord_labels = np.zeros((n_samples, seq_len, chord_labels.shape[0]))


    val_shaped_data = np.zeros((val_n_samples, seq_len, n_features))
    val_shaped_root_labels = np.zeros((val_n_samples, seq_len, root_labels.shape[0]))
    val_shaped_notes_labels = np.zeros((val_n_samples, seq_len, notes_labels.shape[0]))
    val_shaped_beats_labels = np.zeros((val_n_samples, seq_len, beats_labels.shape[0]))
    val_shaped_chord_labels = np.zeros((val_n_samples, seq_len, chord_labels.shape[0]))


    cont = 0
    for i in ind:
        print('Cargando Datos {}/{}'.format(cont,n_samples), end='\r')
        shaped_data[cont,:,:] = all_data[:,i:i+seq_len].T
        shaped_root_labels[cont,:,:] = root_labels[:,i:i+seq_len].T
        shaped_notes_labels[cont,:,:] = notes_labels[:,i:i+seq_len].T
        shaped_beats_labels[cont,:,:] = beats_labels[:,i:i+seq_len].T
        shaped_chord_labels[cont,:,:] = chord_labels[:,i:i+seq_len].T
        cont += 1

    cont = 0
    for i in val_ind:
        print('Cargando Datos {}/{}'.format(cont,val_n_samples), end='\r')
        val_shaped_data[cont,:,:] = val_all_data[:,i:i+seq_len].T
        val_shaped_root_labels[cont,:,:] = val_root_labels[:,i:i+seq_len].T
        val_shaped_notes_labels[cont,:,:] = val_notes_labels[:,i:i+seq_len].T
        val_shaped_beats_labels[cont,:,:] = val_beats_labels[:,i:i+seq_len].T
        val_shaped_chord_labels[cont,:,:] = val_chord_labels[:,i:i+seq_len].T
        cont += 1


    print('train_d: {}, val_d: {}'.format(shaped_data.shape, val_shaped_data.shape))
    print('train_r: {}, val_r: {}'.format(shaped_root_labels.shape, val_shaped_root_labels.shape))
    print('train_n: {}, val_n: {}'.format(shaped_notes_labels.shape, val_shaped_notes_labels.shape))
    print('train_b: {}, val_b: {}'.format(shaped_beats_labels.shape, val_shaped_beats_labels.shape))
    print('train_chord: {}, val_chord: {}'.format(shaped_chord_labels.shape, val_shaped_chord_labels.shape))

    # n_train = shaped_data.shape[0]
    # n_val = val_shaped_data.shape[0]

    _makeTempDir(out_dir)
    for i, data in enumerate(zip(shaped_data, shaped_root_labels, shaped_notes_labels, shaped_beats_labels, shaped_chord_labels)):
        t_d, t_r, t_n, t_b, t_c = data
        curr_name = 'train_sample_' + str(i) + '.pkl'
        curr_data = (t_d, t_r, t_n, t_b, t_c)
        _save_pkl(curr_data, out_dir + '/' + curr_name)

    for i, data in enumerate(zip(val_shaped_data, val_shaped_root_labels, val_shaped_notes_labels, val_shaped_beats_labels, val_shaped_chord_labels)):
        t_d, t_r, t_n, t_b, t_c = data
        curr_name = 'validation_sample_' + str(i) + '.pkl'
        curr_data = (t_d, t_r, t_n, t_b, t_c)
        _save_pkl(curr_data, out_dir + curr_name)

    # return mean, std


def _makeTempDir(dirName):
    if os.path.isdir(dirName):
        return None
    command = 'mkdir ' + dirName
    os.system(command)
    assert os.path.isdir(dirName)

def _open_pickle(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def _save_pkl(data, filename):
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()

def _update_array(array, new_values, axis=1):
    if array is None:
        return new_values
    else:
        array = np.concatenate((array, new_values), axis=axis)
        return array


def encode_labels(gt_root, gt_notes):
    root_ind = {
                    'C':  0,
                    'C#': 1,
                    'C#/Db': 1,
                    'Db': 1,
                    'D':  2,
                    'D#': 3,
                    'D#/Eb': 3,
                    'Eb': 3,
                    'E':  4,
                    'F':  5,
                    'F#': 6,
                    'F#/Gb': 6,
                    'Gb': 6,
                    'G':  7,
                    'G#': 8,
                    'G#/Ab': 8,
                    'Ab': 8,
                    'A':  9,
                    'A#': 10,
                    'A#/Bb': 10,
                    'Bb': 10,
                    'B':  11,
                    'B/Cb':  11,
                    'Cb': 11
                }
    chord_ind = {
                    'maj':  0,
                    '7':    1,
                    'min':  2,
                    'hdim': 3,
                    'dim':  4
                 }

    chord_classes = len(chord_ind.keys())
    class_numbers = []
    for root, chord in zip(gt_root, gt_notes):
        if 'N' in root:
            class_numbers.append(60)
        else:
            curr_class = chord_classes*root_ind[root] + chord_ind[chord]
            class_numbers.append(curr_class)
    return class_numbers


def decoder_split_data_(path, seq_len=100, out_dir='./temp_data_decoder/'):


    labels = None
    all_data = None
    val_labels = None
    val_all_data = None

    # shape is features x len_seq
    all_files = glob.glob(path+'*pkl')
    song_names = set()

    for curr_file in all_files:
        song_names.add(curr_file.split('2048')[0])

    validation_songs = random.sample(song_names, 3)

    for curr_song in all_files:

        data = _open_pickle(curr_song)
        r_pred, n_pred, b_pred = data['latent_features']
        gt_roots, gt_chords = data['name_labels']

        curr_labels = to_categorical(encode_labels(gt_roots, gt_chords),num_classes=61)
        curr_labels = curr_labels.reshape((1, *curr_labels.shape))
        curr_data = np.concatenate((r_pred, n_pred, b_pred), axis=2)

        if curr_song.split('2048')[0] in validation_songs:
            val_labels = _update_array(val_labels, curr_labels)
            val_all_data = _update_array(val_all_data, curr_data)

        else:
            labels = _update_array(labels, curr_labels)
            all_data = _update_array(all_data, curr_data)

    print('Loaded: ')
    print('  labels: {}, val: {}'.format(labels.shape, val_labels.shape))
    print('  all_data: {}, val {}'.format(all_data.shape, val_all_data.shape))

    labels = np.squeeze(labels, axis=0)
    all_data = np.squeeze(all_data, axis=0)
    val_labels = np.squeeze(val_labels, axis=0)
    val_all_data = np.squeeze(val_all_data, axis=0)


    ind = np.arange(0, all_data.shape[0] - seq_len, int(seq_len/5))
    n_samples = len(ind)
    val_ind = np.arange(0, val_all_data.shape[0] - seq_len, int(seq_len/5))
    val_n_samples = len(val_ind)

    n_features = all_data.shape[1]

    shaped_data = np.zeros((n_samples, seq_len, n_features))
    shaped_labels = np.zeros((n_samples, seq_len, labels.shape[1]))

    val_shaped_data = np.zeros((val_n_samples, seq_len, n_features))
    val_shaped_labels = np.zeros((val_n_samples, seq_len, val_labels.shape[1]))

    cont = 0
    for i in ind:
        print('Cargando Datos {}/{}'.format(cont,n_samples), end='\r')
        shaped_data[cont,:,:] = all_data[i:i+seq_len, :]
        shaped_labels[cont,:,:] = labels[i:i+seq_len, :]
        cont += 1

    cont = 0
    for i in val_ind:
        print('Cargando Datos {}/{}'.format(cont,n_samples), end='\r')
        val_shaped_data[cont,:,:] = val_all_data[i:i+seq_len, :]
        val_shaped_labels[cont,:,:] = val_labels[i:i+seq_len, :]
        cont += 1

    print('train_d: {}, val_d: {}'.format(shaped_data.shape, val_shaped_data.shape))
    print('train_l: {}, val_l: {}'.format(shaped_labels.shape, val_shaped_labels.shape))

    n_train = shaped_data.shape[0]
    n_val = val_shaped_data.shape[0]

    _makeTempDir(out_dir)
    for i, data in enumerate(zip(shaped_data, shaped_labels)):
        t_d, t_l = data
        curr_name = 'train_sample_' + str(i) + '.pkl'
        curr_data = (t_d, t_l)
        _save_pkl(curr_data, out_dir + '/' + curr_name)

    for i, data in enumerate(zip(val_shaped_data, val_shaped_labels)):
        t_d, t_l = data
        curr_name = 'validation_sample_' + str(i) + '.pkl'
        curr_data = (t_d, t_l)
        _save_pkl(curr_data, out_dir + curr_name)

    return n_train, n_val

def decoder_split_data(path, out_dir='./temp_data_decoder/'):

    _makeTempDir(out_dir)
    all_files = glob.glob(path+'*pkl')
    song_names = set()

    for curr_file in all_files:
        song_names.add(curr_file.split('2048')[0])

    validation_songs = random.sample(song_names, 3)

    for curr_song in song_names:
        all_choruses = None
        song_name = curr_song.split('/')[-1]
        # print(song_name)

        for i in range(200):
            # print('    i: ' + str(i))
            curr_name = curr_song + '2048_' + str(i)+ '.pkl'
            if not os.path.isfile(curr_name):
                break
            curr_data = _open_pickle(curr_name)
            r_pred, n_pred, b_pred = curr_data['latent_features']
            gt_roots, gt_chords = curr_data['name_labels']
            class_labels = encode_labels(gt_roots, gt_chords)
            beat_labels =  curr_data['labels'][2]
            # print('       latent_features: {}'.format(r_pred.shape))
            if all_choruses is None:
                all_choruses = np.concatenate((r_pred, n_pred, b_pred), axis=2)
            else:
                curr_choruses = np.concatenate((r_pred, n_pred, b_pred), axis=2)
                all_choruses = np.concatenate((all_choruses, curr_choruses), axis=0)
            # print('        all_choruses: {}'.format(all_choruses.shape))

        averaged_choruses = np.mean(all_choruses, axis=0).reshape((1, all_choruses.shape[1], all_choruses.shape[2]))
        class_labels = to_categorical(class_labels, num_classes=61)
        class_labels  = class_labels.reshape((1, *class_labels.shape))
        beat_labels = beat_labels.reshape((1, * beat_labels.shape))
        # print('    averaged_choruses: {}'.format(averaged_choruses.shape))

        data = (averaged_choruses, class_labels, beat_labels)
        if curr_song in validation_songs:
            filename = out_dir + song_name + '2048_validation.pkl'
        else:
            filename = out_dir + song_name + '2048_train.pkl'
        _save_pkl(data, filename)


class chordEval:
    def __init__(self, ):
        self.roots = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G',
                      'G#/Ab', 'A', 'A#/Bb', 'B/Cb', 'N']

        self.chord_types = ['maj', '7', 'min', 'hdim', 'dim']

        self.root_coding =  deque([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Relative to C
        self.type_template = {
                                'maj' : deque([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
                                '7'   : deque([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
                                'min' : deque([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
                                'hdim': deque([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]),
                                'dim' : deque([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
                            }

        self.rotation_index = {
                                'C':  0,
                                'C#': 1,
                                'C#/Db': 1,
                                'Db': 1,
                                'D':  2,
                                'D#': 3,
                                'D#/Eb': 3,
                                'Eb': 3,
                                'E':  4,
                                'F':  5,
                                'F#': 6,
                                'F#/Gb': 6,
                                'Gb': 6,
                                'G':  7,
                                'G#': 8,
                                'G#/Ab': 8,
                                'Ab': 8,
                                'A':  9,
                                'A#': 10,
                                'A#/Bb': 10,
                                'Bb': 10,
                                'B':  11,
                                'B/Cb':  11,
                                'Cb': 11
                            }

        self.root_map = {
                            0:'C',
                            1:'C#/Db',
                            2:'D',
                            3:'D#/Eb',
                            4:'E',
                            5:'F',
                            6:'F#/Gb',
                            7:'G',
                            8:'G#/Ab',
                            9:'A',
                            10:'A#/Bb',
                            11:'B/Cb',
                            12:'N'
                        }
        self.notes_map = {
                            'maj':  0,
                            '7':    1,
                            'min':  2,
                            'hdim': 3,
                            'dim':  4
                         }

    def beat_predictions(self, p_beat):
        beats = np.zeros((p_beat.shape[1]))
        if len(p_beat.shape) > 2:
            p_beat = np.squeeze(p_beat, axis=0)
        for i in range(len(beats)):
            beats[i] = np.argmax(p_beat[i ,:])

        return beats

    def root_predictions(self, p_root):
        if len(p_root.shape) > 2:
            p_root = np.squeeze(p_root, axis=0)
        a,b = p_root.shape
        if a < b:
            p_root = p_root.T

        roots = np.zeros((np.max(p_root.shape)))

        for i in range(len(roots)):
            roots[i] = np.argmax(p_root[i ,:])
        return roots

    def root_name(self, coded_roots):
        roots = []
        for ind in coded_roots:
            roots.append(self.root_map[ind])
        return roots

    def chord_type_greedy(self, roots, notes, thres=0.2, interm=False):
        if len(notes.shape) > 2:
            notes = np.squeeze(notes, axis=0)
        else:
            notes = notes.T

        notes[notes < thres] = 0
        chord_types = []
        type_names = []

        for root, curr_notes in zip(roots, notes):
            root = int(root)

            curr_name, curr_type = self._get_expected_chord(root, curr_notes)
            chord_types.append(curr_type)
            type_names.append(curr_name)
            # print('curr root: {}, curr type: {}'.format(root, curr_name))

        return type_names, chord_types

    def _get_expected_chord(self, root, notes, interm=False):
        # Gets the chord type that maximizes the prob f the predicted root

        if root == 12:
            #print('Max is 12!: {}'.format(notes))
            return 'None', 5

        used_notes = notes#[:-1]

        curr_max = 0
        curr_type_name = ''
        curr_type = None

        for i, itms in enumerate(self.type_template.items()):
            type, content = itms
            content = content.copy()
            content.rotate(root)
            curr_value = np.dot(np.array(list(content)), used_notes)

            if curr_value > curr_max:
                curr_max = curr_value
                curr_type_name = type
                curr_type = i

        if interm:
            return curr_type_name, curr_type, curr_max
        else:
            return curr_type_name, curr_type


    def _get_expected_root_chords(self, roots, notes):
        # Get the combination root-chord type that maximizes the probability.

        # if np.argmax(notes) == 12:
        #     #print('Max is 12!: {}'.format(notes))
        #     return 12, 'None', 5

        max_val = 0
        curr_info = None
        for root in range(12):
            type_name, curr_type, curr_val = self._get_expected_chord(root, notes,
                                                                    interm=True)
            curr_val *= roots[root]
            if curr_val > max_val:
                max_val = curr_val
                curr_info = (root, type_name, curr_type)

        return curr_info

    def chord_type_all(self, roots,  notes, thres=0.2):
        # Get the sequence of predicted root-chors type that maximizes probability

        if len(notes.shape) > 2:
            notes = np.squeeze(notes, axis=0)
        else:
            notes = notes.T
        roots = np.squeeze(roots, axis=0)

        notes[notes < thres] = 0

        pred_roots = []
        names = []
        types = []

        for pred_root, curr_notes in zip(roots, notes):
            curr_root, curr_name, curr_type = self._get_expected_root_chords(pred_root, curr_notes)
            pred_roots.append(curr_root)
            names.append(curr_name)
            types.append(curr_type)

        return pred_roots, names, types


    def get_conf_matrix(self, true, pred, labels):
        if isinstance(labels, tuple):
            roots = labels[0]
            chords = labels[1]
            labels = []
            for root in roots[:-1]:
                for chord in chords:
                    labels.append(root+chord)
            labels.append('N')

        matrix = confusion_matrix(true, pred, labels=labels, normalize='true')
        matrix = np.around(np.multiply(matrix, 100), decimals=2)
        df_roots = pd.DataFrame(matrix, columns=labels, index = labels)
        df_roots.index.name = 'Ground Truth'
        df_roots.columns.name = 'Predicted'
        return df_roots

    def to_chord_label(self, labels):
        label_map = {}
        i = 0
        for root in self.roots[:-1]:
            for chord in self.chord_types:
                label_map[i] = root+':'+chord
                i += 1
        label_map[i] = 'N:N'

        out_labels = []

        for label in labels:
            out_labels.append(label_map[label])

        return out_labels

    def from_categorical(self, array):
        output = []
        for i in array:
            output.append(np.argmax(i))
        return output


    def compareRoot(self, gt, pred):
        """
            Given as root:type strings
        """

        gt_root = [x.split(':')[0] for x in gt]
        pred_root = [x.split(':')[0] for x in pred]

        root_acc = accuracy_score(gt_root, pred_root)
        return root_acc

    def compareMajmin(self, gt, pred):
        pass


    def compareThirds(self, gt, pred):
        """
            Given as root:type strings
        """

        comp_list = []

        for g, p in zip(gt, pred):
            gt_root, gt_type = g.split(':')
            pred_root, pred_type = p.split(':')

            if gt_root == 'N' or pred_root == 'N':
                continue

            elif gt_root != pred_root:
                comp_list.append(0)
            else:
                if self._thirdType(gt_type) == self._thirdType(pred_type):
                    comp_list.append(1)
                else:
                    comp_list.append(0)

        return sum(comp_list)/len(comp_list)

    def _thirdType(self, chord_type):
        if chord_type == 'maj' or chord_type == '7':
            return 'maj'
        else:
            return 'min'

    def compareMirex(self, gt, pred):
        """
            Given as root:type strings. MIREX based evaluation, if they share
            three pitch classes, are considered equal.
        """

        comp_list = []

        for g, p in zip(gt, pred):
            gt_root, gt_type = g.split(':')
            pred_root, pred_type = p.split(':')

            if gt_root == 'N' and pred_root == 'N':
                comp_list.append(1)
                continue
            elif gt_root == 'N' or pred_root == 'N':
                comp_list.append(0)
                continue

            gt_ind = self.rotation_index[gt_root]
            pred_ind = self.rotation_index[pred_root]

            gt_notes = self.type_template[gt_type].copy()
            pred_notes = self.type_template[pred_type].copy()

            gt_notes.rotate(gt_ind)
            pred_notes.rotate(pred_ind)

            sim = np.dot(np.array(list(gt_notes)), np.array(list(pred_notes)))
            if sim >= 3:
                comp_list.append(1)
            else:
                comp_list.append(0)


            # print('gt: {}, pred: {}  --   {}'.format(g, p, comp_list[-1]))
            # print('    notes gt:   {}'.format(list(gt_notes)))
            # print('    notes pred: {}'.format(list(pred_notes)))
            # print(' ')


        return sum(comp_list)/len(comp_list)




class chordTesting:

    def __init__(self, type):
        """
            Type hast to be one of ['all', 'average_cqt',
                                    'average_latent_features',
                                    'average_predictions]
        """

        self.gt_root = []
        self.gt_notes = []
        self.gt_beats = []
        self.gt_chords = []

        self.pred_root = []
        self.pred_notes = []
        self.pred_beats = []
        self.pred_chords = []

        self.type = type

        self.stack_hidden = None
        self.stack_root = None
        self.stack_notes = None
        self.stack_beats = None
        self.stack_chords = None

        self.concatenatedLatent = None

    def updateValues(self, new_root, new_notes, new_beats, new_chords, data_type):
        """
            Concatenate new values into a single array.
        """
        if data_type == 'gt':
            self.gt_root.extend(new_root)
            self.gt_notes.extend(new_notes)
            self.gt_beats.extend(new_beats)
            self.gt_chords.extend(new_chords)

        elif data_type == 'pred':
            self.pred_root.extend(new_root)
            self.pred_notes.extend(new_notes)
            self.pred_beats.extend(new_beats)
            self.pred_chords.extend(new_chords)

    def stackLatentValues(self, new_hidden, new_root, new_notes, new_beats):
        """
            Use only to store latent features, not final predictions
        """

        self.stack_hidden = _update_array(self.stack_hidden, new_hidden, axis=0)
        self.stack_root = _update_array(self.stack_root, new_root, axis=0)
        self.stack_notes = _update_array(self.stack_notes, new_notes, axis=0)
        self.stack_beats = _update_array(self.stack_beats, new_beats, axis=0)


    def averageLatentStack(self):
        """
            Use only to store latent features, not final predictions
        """

        self.stack_hidden = np.mean(self.stack_hidden, axis=0)
        self.stack_hidden = self.stack_hidden.reshape((1, *self.stack_hidden.shape))
        self.stack_root = np.mean(self.stack_root, axis=0)
        self.stack_root = self.stack_root.reshape((1, *self.stack_root.shape))
        self.stack_notes = np.mean(self.stack_notes, axis=0)
        self.stack_notes = self.stack_notes.reshape((1, *self.stack_notes.shape))
        self.stack_beats = np.mean(self.stack_beats, axis=0)
        self.stack_beats = self.stack_beats.reshape((1, *self.stack_beats.shape))


        self.concatenatedLatent = np.concatenate((self.stack_hidden,
                                                  self.stack_root,
                                                  self.stack_notes), axis=2)

    def stackPredValues(self,new_root, new_notes, new_beats, new_chords):
        """
            Use only to store final predictions, not latent features.
        """

        # if self.stack_root is not None:
        #     print('r_pred_stack {} n_pred_stack {} b_pred_stack {} c_pred_stack {}'.format(self.stack_root.shape, self.stack_notes.shape, self.stack_beats.shape, self.stack_chords.shape))

        self.stack_root = _update_array(self.stack_root, new_root, axis=0)
        self.stack_notes = _update_array(self.stack_notes, new_notes, axis=0)
        self.stack_beats = _update_array(self.stack_beats, new_beats, axis=0)
        self.stack_chords = _update_array(self.stack_chords, new_chords, axis=0)


    def averagePredStack(self):
        """
            Use only to store final predictions, not latent features.
        """

        self.stack_root = np.mean(self.stack_root, axis=0)
        self.stack_root = self.stack_root.reshape((1, *self.stack_root.shape))
        self.stack_notes = np.mean(self.stack_notes, axis=0)
        self.stack_notes = self.stack_notes.reshape((1, *self.stack_notes.shape))
        self.stack_beats = np.mean(self.stack_beats, axis=0)
        self.stack_beats = self.stack_beats.reshape((1, *self.stack_beats.shape))
        self.stack_chords = np.mean(self.stack_chords, axis=0)
        self.stack_chords = self.stack_chords.reshape((1, *self.stack_chords.shape))


    def resetStack(self):
        self.stack_hidden = None
        self.stack_root = None
        self.stack_notes = None
        self.stack_beats = None
        self.stack_chords = None
        self.concatenatedLatent = None


    @staticmethod
    def toHRlabels(r_pred, n_pred, b_pred, c_pred, CE):
        pred_beats = CE.beat_predictions(b_pred).tolist() # Pred beats
        pred_roots_interm = CE.root_predictions(r_pred)
        pred_roots = CE.root_name(pred_roots_interm) # Pred root
        pred_notes, _ = CE.chord_type_greedy(pred_roots_interm, n_pred) # Pred Chord Type
        pred_chords_list = np.squeeze(c_pred,axis=0).tolist()
        pred_chords = CE.to_chord_label(CE.from_categorical(pred_chords_list))

        return pred_roots, pred_notes, pred_beats, pred_chords












#
