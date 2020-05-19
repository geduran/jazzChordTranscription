import os
import sys
import glob
import pickle
import numpy         as     np
import tensorflow    as     tf
import matplotlib.pyplot as plt
from   keras         import backend as K
from sklearn.metrics import accuracy_score
from   keras.backend import tensorflow_backend
from   keras.utils.np_utils import to_categorical
from   dlUtils       import functional_decoder, encode_labels
from   dlUtils       import chordEval, _open_pickle

# tensorflow configuration
K.set_image_dim_ordering('th')
print(K.image_data_format())
config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

method = 'cqt'
model_paths = '../../models/chordTranscription/decoder/'
test_data_path = '../../data/JAAH/chordTranscription/decoder/' + method + '/test/'
results_path = '../../results/chordTranscription/decoder/'

model_name    = 'decoder_best.h5'


epochs         = 40
n_hidden       = 64
# seq_len        = 50
num_features   = 26
n_labels       = 61

input_shape  = (None, num_features)


model = functional_decoder(input_shape)
model.load_weights(model_paths + model_name)
model.summary()

CE = chordEval()

test_songs = glob.glob(test_data_path + '*.pkl')

song_names = set()

all_gt = []
all_gt_averaged = []
all_predictions = []
all_predictions_averaged = []


for curr_file in test_songs:
    song_names.add(curr_file.split('2048')[0])

for curr_song in song_names:
    all_choruses = None
    song_name = curr_song.split('/')[-1]

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
            predictions = model.predict(all_choruses, batch_size=32, verbose=0)
        else:
            curr_choruses = np.concatenate((r_pred, n_pred), axis=2)
            predictions = model.predict(curr_choruses, batch_size=32, verbose=0)
            all_choruses = np.concatenate((all_choruses, curr_choruses), axis=0)
        # print('        all_choruses: {}'.format(all_choruses.shape))

        all_gt.extend(class_labels)
        all_predictions.extend(np.squeeze(predictions,axis=0).tolist())

    averaged_choruses = np.mean(all_choruses, axis=0).reshape((1, all_choruses.shape[1], all_choruses.shape[2]))
    # class_labels = to_categorical(class_labels, num_classes=61)

    predictions = model.predict(averaged_choruses, batch_size=32, verbose=0)

    all_gt_averaged.extend(class_labels)
    all_predictions_averaged.extend(np.squeeze(predictions,axis=0).tolist())

    # print('curr_gt: {}, all_gt: {}'.format(class_labels.shape, len(all_gt)))
    # print('predictions: {}, all_predictions: {}'.format(predictions.shape, len(all_predictions)))


all_predictions = CE.from_categorical(all_predictions)
acc = accuracy_score(all_gt, all_predictions)

all_predictions_averaged = CE.from_categorical(all_predictions_averaged)
acc_averaged = accuracy_score(all_gt_averaged, all_predictions_averaged)

print('Acc: {0:.2f}%'.format(acc*100))
print('Acc averaged: {0:.2f}%'.format(acc_averaged*100))





#
