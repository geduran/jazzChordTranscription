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



class DataGenerator(keras.utils.Sequence):
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
            monitor        = 'root_acc',
            patience       = 10,
            mode           = 'max',
            verbose        = 1),
        ModelCheckpoint(model_file,
            monitor        = 'root_acc',
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



def split_data(path, averaged=False, seq_len=100, out_dir='./temp_data/'):


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

    train_d, val_d, train_r, val_r, train_n, val_n, train_b, val_b = split_data

    print('train_d: {}, val_d: {}'.format(train_d.shape, val_d.shape))
    print('train_r: {}, val_r: {}'.format(train_r.shape, val_r.shape))
    print('train_n: {}, val_n: {}'.format(train_n.shape, val_n.shape))
    print('train_b: {}, val_b: {}'.format(train_b.shape, val_b.shape))

    n_train = train_d.shape[0]
    n_val = val_d.shape[0]

    _makeTempDir(out_dir)
    for i, data in enumerate(zip(train_d, train_r, train_n, train_b)):
        t_d, t_r, t_n, t_b = data
        curr_name = 'train_sample_' + str(i) + '.pkl'
        curr_data = (t_d, t_r, t_n, t_b)
        _save_pkl(curr_data, out_dir + '/' + curr_name)

    for i, data in enumerate(zip(val_d, val_r, val_n, val_b)):
        t_d, t_r, t_n, t_b = data
        curr_name = 'validation_sample_' + str(i) + '.pkl'
        curr_data = (t_d, t_r, t_n, t_b)
        _save_pkl(curr_data, out_dir + curr_name)

    return n_train, n_val

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




#
# input_shape = (None, 84, 1)
# functional_encoder(input_shape)
#
