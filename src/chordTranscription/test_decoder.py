import os
import sys
import glob
import pickle
import numpy         as     np
import tensorflow    as     tf
import matplotlib.pyplot as plt
import seaborn       as     sn
import pandas        as     pd
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
# model.summary()

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
all_predictions = CE.to_chord_label(all_predictions)
all_gt = CE.to_chord_label(all_gt)
acc = accuracy_score(all_gt, all_predictions)

all_predictions_averaged = CE.from_categorical(all_predictions_averaged)
all_predictions_averaged = CE.to_chord_label(all_predictions_averaged)
all_gt_averaged = CE.to_chord_label(all_gt_averaged)
acc_averaged = accuracy_score(all_gt_averaged, all_predictions_averaged)

results_name = model_name[:-3] + '_results.txt'

with open(results_path + results_name  , 'w') as f:
    f.write('\nFor Model ' + results_name + '\n')
    f.write('    Acc: {0:.2f}%'.format(acc*100) + '\n')
    f.write('    Acc averaged: {0:.2f}%'.format(acc_averaged*100))

with open(results_path + results_name , 'r') as f:
    for line in f:
        print(line)



matrix_name = model_name[:-3] + '_averaged_roots_conf.eps'
av_gt_roots = [x.split(':')[0] for x in all_gt_averaged]
av_pred_roots = [x.split(':')[0] for x in all_predictions_averaged]
df_roots = CE.get_conf_matrix(av_gt_roots, av_pred_roots, labels=CE.roots)
plt.figure(figsize = (10,7))
# sn.set(font_scale=0.8)#for label size
sn.heatmap(df_roots, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.savefig(results_path + matrix_name, format='eps', dpi=50)

plt.clf()

matrix_name = model_name[:-3] + '_averaged_type_conf.eps'
av_gt_chords = [x.split(':')[1] for x in all_gt_averaged]
av_pred_chords = [x.split(':')[1] for x in all_predictions_averaged]
df_chords = CE.get_conf_matrix(av_gt_chords, av_pred_chords, labels=CE.chord_types)
plt.figure(figsize = (10,7))
# sn.set(font_scale=0.8)#for label size
sn.heatmap(df_chords, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.savefig(results_path + matrix_name, format='eps', dpi=50)


##############################################################################

matrix_name = model_name[:-3] + '_roots_conf.eps'
gt_roots = [x.split(':')[0] for x in all_gt]
pred_roots = [x.split(':')[0] for x in all_predictions]
df_roots = CE.get_conf_matrix(gt_roots, pred_roots, labels=CE.roots)
plt.figure(figsize = (10,7))
# sn.set(font_scale=0.8)#for label size
sn.heatmap(df_roots, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.savefig(results_path + matrix_name, format='eps', dpi=50)

plt.clf()

matrix_name = model_name[:-3] + '_type_conf.eps'
gt_chords = [x.split(':')[1] for x in all_gt]
pred_chords = [x.split(':')[1] for x in all_predictions]
df_chords = CE.get_conf_matrix(gt_chords, pred_chords, labels=CE.chord_types)
plt.figure(figsize = (10,7))
# sn.set(font_scale=0.8)#for label size
sn.heatmap(df_chords, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.savefig(results_path + matrix_name, format='eps', dpi=50)




#
