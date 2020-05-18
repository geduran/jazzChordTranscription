import numpy                as np
import keras
from   keras.models         import Model, Sequential
from   keras.layers         import Dense, Conv2D, Flatten, Activation, Dropout
from   keras.layers         import regularizers, MaxPooling2D, Lambda
from   keras.layers         import GRU, LSTM, SimpleRNN, BatchNormalization
from   keras.layers         import Bidirectional, TimeDistributed, Input
from   keras.callbacks      import ModelCheckpoint, EarlyStopping, Callback
from   keras.utils.np_utils import to_categorical
import keras.backend as K
import tensorflow as tf
from   sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
import sklearn.model_selection
from   collections import deque
import random
import pickle
import glob
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
        return self._n_samples

    def __getitem__(self, index):
        'Generate one batch of data'

        # Current index
        curr_index = self.indexes[index]

        # Generate data
        X, labels = self.__data_generation(self.list_IDs[curr_index])

        return X, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ID_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Generate data
        with open(ID_temp, 'rb') as f:
            X, Y = pickle.load(f)

        # X = X.reshape((*X.shape, 1))
        return X, Y



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

    # For root and notes
    conv4 = Conv2D(42, (1, int(conv1.shape[2])), padding='valid',
                   activation='relu', data_format='channels_last')(conv3_bn)

    conv4_bn = BatchNormalization()(conv4)

    sq_conv4 = Lambda(lambda x: K.squeeze(x, axis=2))(conv4_bn)

    gru1 = Bidirectional(GRU(64, return_sequences=True))(sq_conv4)

    root_output = TimeDistributed(Dense(13, activation='softmax'), name='root')(gru1)

    notes_output = TimeDistributed(Dense(13, activation='sigmoid'), name='notes')(gru1)

    # For beats
    conv5 = Conv2D(1,  kernel_size=(7, 3), padding='same',
                   activation='relu', data_format='channels_last')(conv3_bn)

    conv5_bn = BatchNormalization()(conv5)

    # conv5_mp = MaxPooling2D(pool_size=(1,2), data_format='channels_last')(conv5_bn)

    sq_conv5 = Lambda(lambda x: K.squeeze(x, axis=3))(conv5_bn)

    gru2 = Bidirectional(GRU(64, return_sequences=True))(sq_conv5)

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

    # model.summary()
    return model


def functional_decoder(input_shape):

    RNN_type = GRU

    #Start Neural Network
    model = Sequential()

    model.add(Bidirectional(RNN_type(124, return_sequences=True),
                            input_shape=input_shape))

    model.add(Bidirectional(RNN_type(124, return_sequences=True)))

    # model.add(RNN_type(n_hidden, return_sequences=True,
    #           input_shape=input_shape))
    # model.add(RNN_type(n_hidden, return_sequences=True))
    # model.add(Dropout(0.5))


    model.add(TimeDistributed(Dense(61, activation='softmax')))


    model.compile(loss        = keras.losses.categorical_crossentropy,
                  optimizer   = 'adam',
                  metrics     = ['accuracy'])

#    model.summary()

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



def encoder_split_data(path, averaged=False, seq_len=100, out_dir='./temp_data_encoder/'):


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

def decoder_split_data(path, out_dir='./temp_data_decoder/'):

    _makeTempDir(out_dir)
    all_files = glob.glob(path+'*pkl')
    song_names = set()

    for curr_file in all_files:
        song_names.add(curr_file.split('2048')[0])

    validation_songs = random.sample(song_names, 2)

    for curr_song in song_names:
        all_choruses = None
        song_name = curr_song.split('/')[-1]
        print(song_name)

        for i in range(200):
            # print('    i: ' + str(i))
            curr_name = curr_song + '2048_' + str(i)+ '.pkl'
            if not os.path.isfile(curr_name):
                break
            curr_data = _open_pickle(curr_name)
            r_pred, n_pred, b_pred = curr_data['latent_features']
            gt_roots, gt_chords = curr_data['name_labels']
            class_labels = encode_labels(gt_roots, gt_chords)
            # print('       latent_features: {}'.format(r_pred.shape))
            if all_choruses is None:
                all_choruses = np.concatenate((r_pred, n_pred), axis=2)
            else:
                curr_choruses = np.concatenate((r_pred, n_pred), axis=2)
                all_choruses = np.concatenate((all_choruses, curr_choruses), axis=0)
            # print('        all_choruses: {}'.format(all_choruses.shape))

        averaged_choruses = np.mean(all_choruses, axis=0).reshape((1, all_choruses.shape[1], all_choruses.shape[2]))
        class_labels = to_categorical(class_labels, num_classes=61)
        class_labels  = class_labels.reshape((1, *class_labels.shape))
        # print('    averaged_choruses: {}'.format(averaged_choruses.shape))

        data = (averaged_choruses, class_labels)
        if curr_song in validation_songs:
            filename = out_dir + song_name + '2048_validation.pkl'
        else:
            filename = out_dir + song_name + '2048_train.pkl'
        _save_pkl(data, filename)


class chordEval:
    def __init__(self, ):
        self.root_coding =  deque([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Relative to C
        self.type_template = {
                                'maj' : deque([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
                                '7'   : deque([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
                                'min' : deque([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
                                'hdim': deque([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]),
                                'dim' : deque([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
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

        if np.argmax(notes) == 12:
            #print('Max is 12!: {}'.format(notes))
            return 'None', 5

        used_notes = notes[:-1]

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

        if np.argmax(notes) == 12:
            #print('Max is 12!: {}'.format(notes))
            return 12, 'None', 5
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
# input_shape = (None, 84, 1)
# functional_encoder(input_shape)
#
