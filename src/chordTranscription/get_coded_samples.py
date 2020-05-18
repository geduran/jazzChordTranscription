import sys
import glob
import pickle
import numpy         as     np
import tensorflow    as     tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from   keras         import backend as K
from   keras.backend import tensorflow_backend
from   keras.utils.np_utils import to_categorical
from   dlUtils       import functional_encoder
from   dlUtils       import chordEval

# tensorflow configuration
K.set_image_dim_ordering('th')
print(K.image_data_format())
config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

method = 'cqt'
model_paths = '../../models/chordTranscription/encoder/'
train_data_path  = '../../data/JAAH/chordTranscription/encoder/' + method + '/'
interm_data_path = '../../data/JAAH/chordTranscription/intermediate/' + method + '/'

model_name    = 'encoder_best.h5'

epochs         = 20
n_hidden       = 60
batch_size     = 128
# seq_len        = 100
num_features   = 84
n_labels_root  = 13
n_labels_notes = 13
n_labels_beats = 2
input_shape  = (None, num_features, 1)


model = functional_encoder(input_shape)
model.load_weights(model_paths + model_name)

CE = chordEval()

train_songs = glob.glob(train_data_path + '*.pkl')


for curr_song in train_songs:

    with open(curr_song, 'rb') as f:
        data = pickle.load(f)

    all_data = data['data']

    gt_root = data['root_labels']
    gt_notes = data['notes_labels']
    gt_beats = to_categorical(data['beats_labels']).T

    name_root = CE.root_name(CE.root_predictions(gt_root))
    name_chord, _ = CE.chord_type_greedy(CE.root_predictions(gt_root), gt_notes)


    for i_, curr_data in enumerate(all_data):
        print('train Song {}, chorus {}'.format(curr_song.split('/')[-1], i_) , end='                    \r')
        train_data = np.zeros((1, *curr_data.T.shape, 1))
        train_data[0,:,:,0] = curr_data.T

        r_pred, n_pred, b_pred = model.predict(train_data, batch_size=32,
                                    verbose=0)

        file_name = curr_song.split('/')[-1].split('.')[0] + '_' + str(i_) + '.pkl'

        out_data = {'latent_features': (r_pred, n_pred, b_pred),
                    'labels': (gt_root, gt_notes, gt_beats),
                    'name_labels': (name_root, name_chord)}

        f = open(interm_data_path + file_name, 'wb')
        pickle.dump(out_data, f)
        f.close()
