import os
import sys
import glob
import pickle
import numpy         as     np
import tensorflow    as     tf
import matplotlib.pyplot as plt
import seaborn       as     sn
import pandas        as     pd
import keras
from   keras         import backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score
from   keras.utils.np_utils import to_categorical
from   keras.backend import tensorflow_backend
from   dlUtils       import functional_both, encode_labels, only_decoder
from   dlUtils       import chordEval, _open_pickle, chordTesting


# tensorflow configuration
K.set_image_dim_ordering('th')
print(K.image_data_format())
config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


method = 'cqt'
model_paths = '../../models/chordTranscription/both_second/'
test_data_path = '../../data/JAAH/chordTranscription/both_da/' + method + '/test/'
results_path = '../../results/chordTranscription/both/'

if len(sys.argv) > 1:
    model_name = 'both_cqt_' + sys.argv[1] + '.h5'
else:
    model_name    = 'both_second_best.h5'


# epochs         = 40
n_hidden       = 64
# seq_len        = 50
num_features   = 84
n_labels       = 61

input_shape  = (None, num_features, 1)


model = functional_both(input_shape)
model.load_weights(model_paths + model_name)
model.summary()

encoder = keras.Model(inputs=model.input, outputs=
                            [model.get_layer('sq_conv4').output,
                             model.get_layer('root').output,
                             model.get_layer('notes').output,
                             model.get_layer('beats').output])

dec_shape = (None, 109)
decoder = only_decoder(dec_shape)
decoder.load_weights(model_paths + model_name, by_name=True)


CE = chordEval()

test_songs = glob.glob(test_data_path + '*when*0_*.pkl')

testAll = chordTesting('all')
testLatent = chordTesting('average_latent_features')
testPred = chordTesting('average_predictions')



for curr_song in test_songs:
    with open(curr_song, 'rb') as f:
        data = pickle.load(f)

    gt_root_labels = data['root_labels']
    gt_notes_labels = data['notes_labels']

    gt_roots_pred = CE.root_predictions(gt_root_labels)
    gt_roots = CE.root_name(gt_roots_pred) # GT roots

    gt_notes, _ = CE.chord_type_greedy(gt_roots_pred, gt_notes_labels) # GT chord type
    gt_chords_labels = encode_labels(gt_roots, gt_notes)

    all_data = data['data']

    gt_beats = data['beats_labels'].tolist() # GT beats
    gt_chords = CE.to_chord_label(gt_chords_labels) # GT chords

    # Single GT labels
    testLatent.updateValues(gt_roots, gt_notes, gt_beats, gt_chords, 'gt')
    testPred.updateValues(gt_roots, gt_notes, gt_beats, gt_chords, 'gt')


    for i_, curr_data in enumerate(all_data):
        print('test Song {}, chorus {}          '.format(curr_song.split('/')[-1], i_) , end='\r')

        test_data = np.zeros((1, *curr_data.T.shape, 1))
        test_data[0,:,:,0] = curr_data.T

        r_pred, n_pred, b_pred, c_pred = model.predict(test_data, verbose=0)

        all_predictions = chordTesting.toHRlabels(r_pred, n_pred, b_pred, c_pred, CE)
        pred_roots, pred_notes, pred_beats, pred_chords = all_predictions

        # GT labels and prediction for each chorus
        testAll.updateValues(gt_roots, gt_notes, gt_beats, gt_chords, 'gt')
        testAll.updateValues(pred_roots, pred_notes, pred_beats, pred_chords, 'pred')

        # r_pred[r_pred < 0.5] = 0
        # n_pred[n_pred < 0.5] = 0

        # Stack encoded predictions of each chorus
        l_interm, r_interm, n_interm, b_interm = encoder.predict(test_data, verbose=0)
        testLatent.stackLatentValues(l_interm, r_pred, n_pred, b_pred)

        # Stack predictions of each chorus
        testPred.stackPredValues(r_pred, n_pred, b_pred, c_pred)


    # Average predictions and latent features for each song
    testLatent.averageLatentStack()
    testPred.averagePredStack()
    # Obtain chords prediction for averaged latent features
    c_pred_decoder = decoder.predict(testLatent.concatenatedLatent, verbose=0)

    # Stacked predictions
    r_average = testPred.stack_root
    n_average = testPred.stack_notes
    b_average = testPred.stack_beats
    c_average = testPred.stack_chords

    # Human readable labels for averaged predictions
    pred_average_predictions = chordTesting.toHRlabels(r_average, n_average, b_average, c_average, CE)
    latent_average_predictions = chordTesting.toHRlabels(r_average, n_average, b_average, c_pred_decoder, CE)

    # Update of averaged predicted values
    testPred.updateValues(*pred_average_predictions, 'pred')
    testLatent.updateValues(*latent_average_predictions, 'pred')

    # Erase current stack, so the next song can be analized
    testPred.resetStack()
    testLatent.resetStack()


# All choruses accuracy metrics
all_root_acc = accuracy_score(testAll.gt_root, testAll.pred_root)

all_notes_acc = accuracy_score(testAll.gt_notes, testAll.pred_notes)

all_beats_acc = accuracy_score(testAll.gt_beats, testAll.pred_beats)
all_beats_precision = precision_score(testAll.gt_beats, testAll.pred_beats)
all_beats_recall = recall_score(testAll.gt_beats, testAll.pred_beats)

all_chords_acc = accuracy_score(testAll.gt_chords, testAll.pred_chords)
all_chords_root = CE.compareRoot(testAll.gt_chords, testAll.pred_chords)
all_chords_mirex = CE.compareMirex(testAll.gt_chords, testAll.pred_chords)
all_chords_thirds = CE.compareThirds(testAll.gt_chords, testAll.pred_chords)


# Averaged predictions accuracy metrics
avPred_root_acc = accuracy_score(testPred.gt_root, testPred.pred_root)

avPred_notes_acc = accuracy_score(testPred.gt_notes, testPred.pred_notes)

avPred_beats_acc = accuracy_score(testPred.gt_beats, testPred.pred_beats)
avPred_beats_precision = precision_score(testPred.gt_beats, testPred.pred_beats)
avPred_beats_recall = recall_score(testPred.gt_beats, testPred.pred_beats)

avPred_chords_acc = accuracy_score(testPred.gt_chords, testPred.pred_chords)
avPred_chords_root = CE.compareRoot(testPred.gt_chords, testPred.pred_chords)
avPred_chords_mirex = CE.compareMirex(testPred.gt_chords, testPred.pred_chords)
avPred_chords_thirds = CE.compareThirds(testPred.gt_chords, testPred.pred_chords)


# Averaged latent features accuracy metrics
avLat_root_acc = accuracy_score(testLatent.gt_root, testLatent.pred_root)

avLat_notes_acc = accuracy_score(testLatent.gt_notes, testLatent.pred_notes)

avLat_beats_acc = accuracy_score(testLatent.gt_beats, testLatent.pred_beats)
avLat_beats_precision = precision_score(testLatent.gt_beats, testLatent.pred_beats)
avLat_beats_recall = recall_score(testLatent.gt_beats, testLatent.pred_beats)

avLat_chords_acc = accuracy_score(testLatent.gt_chords, testLatent.pred_chords)
avLat_chords_root = CE.compareRoot(testLatent.gt_chords, testLatent.pred_chords)
avLat_chords_mirex = CE.compareMirex(testLatent.gt_chords, testLatent.pred_chords)
avLat_chords_thirds = CE.compareThirds(testLatent.gt_chords, testLatent.pred_chords)



results_name = model_name[:-3] + '_results.txt'

with open(results_path + results_name  , 'w') as f:
    f.write('\nFor Model ' + results_name + '\n')
    f.write('    All choruses' + '\n')
    f.write('        chords_accuracy: {0:.2f}'.format(all_chords_acc) + '\n')
    f.write('        chord_root_accuracy: {0:.2f}'.format(all_chords_root) + '\n')
    f.write('        chords_thirds_accuracy: {0:.2f}'.format(all_chords_thirds) + '\n')
    f.write('        chords_mirex_accuracy: {0:.2f}'.format(all_chords_mirex) + '\n')
    f.write('        root_accuracy: {0:.2f}'.format(all_root_acc) + '\n')
    f.write('        notes_accuracy: {0:.2f}'.format(all_notes_acc) + '\n')
    f.write('        beats_accuracy: {0:.2f}'.format(all_beats_acc) + '\n')
    f.write('        beats_precision: {0:.2f}'.format(all_beats_precision) + '\n')
    f.write('        beats_recall: {0:.2f}'.format(all_beats_recall) + '\n')
    f.write('    Averaged Predictions' + '\n')
    f.write('        chords_accuracy: {0:.2f}'.format(avPred_chords_acc) + '\n')
    f.write('        chord_root_accuracy: {0:.2f}'.format(avPred_chords_root) + '\n')
    f.write('        chords_thirds_accuracy: {0:.2f}'.format(avPred_chords_thirds) + '\n')
    f.write('        chords_mirex_accuracy: {0:.2f}'.format(avPred_chords_mirex) + '\n')
    f.write('        root_accuracy: {0:.2f}'.format(avPred_root_acc) + '\n')
    f.write('        notes_accuracy: {0:.2f}'.format(avPred_notes_acc) + '\n')
    f.write('        beats_accuracy: {0:.2f}'.format(avPred_beats_acc) + '\n')
    f.write('        beats_precision: {0:.2f}'.format(avPred_beats_precision) + '\n')
    f.write('        beats_recall: {0:.2f}'.format(avPred_beats_recall) + '\n')
    f.write('    Averaged Latent Features' + '\n')
    f.write('        chords_accuracy: {0:.2f}'.format(avLat_chords_acc) + '\n')
    f.write('        chord_root_accuracy: {0:.2f}'.format(avLat_chords_root) + '\n')
    f.write('        chords_thirds_accuracy: {0:.2f}'.format(avLat_chords_thirds) + '\n')
    f.write('        chords_mirex_accuracy: {0:.2f}'.format(avLat_chords_mirex) + '\n')



    # f.write('        root_accuracy: {0:.2f}'.format(avLat_root_acc) + '\n')
    # f.write('        notes_accuracy: {0:.2f}'.format(avLat_notes_acc) + '\n')
    # f.write('        beats_accuracy: {0:.2f}'.format(avLat_beats_acc) + '\n')
    # f.write('        beats_precision: {0:.2f}'.format(avLat_beats_precision) + '\n')
    # f.write('        beats_recall: {0:.2f}'.format(avLat_beats_recall) + '\n')



with open(results_path + results_name , 'r') as f:
    for line in f:
        print(line)


#
#
# matrix_name = model_name[:-3] + '_roots_conf.eps'
#
# df_roots = CE.get_conf_matrix(all_gt_roots, all_pred_roots, labels=CE.roots)
# plt.figure(figsize = (10,7))
# # sn.set(font_scale=0.8)#for label size
# sn.heatmap(df_roots, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5
# t -= 0.5
# plt.ylim(b, t)
# plt.savefig(results_path + matrix_name, format='eps', dpi=50)
#
# plt.clf()
#
#
# matrix_name = model_name[:-3] + '_notes_conf.eps'
#
# df_notes = CE.get_conf_matrix(all_gt_notes, all_pred_notes, labels=CE.chord_types)
# plt.figure(figsize = (10,7))
# # sn.set(font_scale=0.8)#for label size
# sn.heatmap(df_notes, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5
# t -= 0.5
# plt.ylim(b, t)
# plt.savefig(results_path + matrix_name, format='eps', dpi=50)
# plt.clf()
#
#
# # (all_gt_chords, all_pred_chords)
#
# matrix_name = model_name[:-3] + '_roots_chord_conf.eps'
#
# gt_roots = [x.split(':')[0] for x in all_gt_chords]
# pred_roots = [x.split(':')[0] for x in all_pred_chords]
# df_roots = CE.get_conf_matrix(gt_roots, pred_roots, labels=CE.roots)
# plt.figure(figsize = (10,7))
# # sn.set(font_scale=0.8)#for label size
# sn.heatmap(df_roots, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5
# t -= 0.5
# plt.ylim(b, t)
# plt.savefig(results_path + matrix_name, format='eps', dpi=50)
#
# plt.clf()
#
# matrix_name = model_name[:-3] + '_type_chord_conf.eps'
#
# gt_chords = [x.split(':')[1] for x in all_gt_chords]
# pred_chords = [x.split(':')[1] for x in all_pred_chords]
# df_chords = CE.get_conf_matrix(gt_chords, pred_chords, labels=CE.chord_types)
# plt.figure(figsize = (10,7))
# # sn.set(font_scale=0.8)#for label size
# sn.heatmap(df_chords, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5
# t -= 0.5
# plt.ylim(b, t)
# plt.savefig(results_path + matrix_name, format='eps', dpi=50)
#         #






# For Model both_cqt_06_results.txt
#
#     All choruses
#
#         chords_accuracy: 0.53
#
#         root_accuracy: 0.60
#
#         notes_accuracy: 0.49
#
#         beats_accuracy: 0.86
#
#         beats_precision: 0.52
#
#         beats_recall: 0.17
#
#     Averaged Predictions
#
#         chords_accuracy: 0.64
#
#         root_accuracy: 0.68
#
#         notes_accuracy: 0.55
#
#         beats_accuracy: 0.87
#
#         beats_precision: 0.82
#
#         beats_recall: 0.01
#
#     Averaged Latent Features
#
#         chords_accuracy: 0.61















  #
