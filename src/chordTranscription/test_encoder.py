import sys
import glob
import pickle
import numpy         as     np
import pandas        as     pd
import seaborn       as     sn
import tensorflow    as     tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
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
test_data_path = '../../data/JAAH/chordTranscription/encoder/' + method + '/test/'
results_path = '../../results/chordTranscription/encoder/'


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

test_songs = glob.glob(test_data_path + '*.pkl')

all_gt_roots = []
all_gt_notes = []
all_gt_beats = []

all_pred_roots = []
all_pred_notes = []
all_pred_roots_greedy = []
all_pred_notes_greedy = []
all_pred_beats = []

average_pred_roots = []
average_pred_notes = []
average_pred_roots_greedy = []
average_pred_notes_greedy = []
average_pred_beats = []

single_gt_roots = []
single_gt_notes = []
single_gt_beats = []

modality = ''
# Modalities can be average_cqt, average_latent_features or empty
if len(sys.argv) == 2:
    modality = sys.argv[1]
    if modality != 'average_cqt' and modality != 'average_latent_features':
        print('Invalid modality...!')
        sys.exit(0)

for curr_song in test_songs:
    with open(curr_song, 'rb') as f:
        data = pickle.load(f)

    gt_root_labels = data['root_labels']
    gt_notes_labels = data['notes_labels']
    gt_beats_labels = to_categorical(data['beat_labels']).T


    if modality == 'average_cqt':
        all_data = [data['averaged_data']]
    else:
        all_data = data['data']
    # all_data.append(av_data)

    song_roots = None
    song_notes = None
    song_beats = None

    for i_, curr_data in enumerate(all_data):
        print('test Song {}, chorus {}'.format(curr_song.split('/')[-1], i_) , end='                    \r')
        test_data = np.zeros((1, *curr_data.T.shape, 1))
        test_data[0,:,:,0] = curr_data.T

        r_pred, n_pred, b_pred = model.predict(test_data, batch_size=32,
                                    verbose=0)

        if song_roots is None:
            song_roots = r_pred
            song_notes = n_pred
            song_beats = b_pred
        else:
            song_roots = np.concatenate((song_roots, r_pred), axis=0)
            song_notes = np.concatenate((song_notes, n_pred), axis=0)
            song_beats = np.concatenate((song_beats, b_pred), axis=0)

        # print('song_roots: {}'.format(song_roots.shape))
        # print('song_notes: {}'.format(song_notes.shape))
        # print('song_beats: {}'.format(song_beats.shape))

        pred_beats = CE.beat_predictions(b_pred)
        all_pred_beats.extend(pred_beats.tolist())

        pred_roots = CE.root_predictions(r_pred)
        gt_roots = CE.root_predictions(gt_root_labels)
        all_pred_roots_greedy.extend(CE.root_name(pred_roots))

        gt_chord_name, gt_chord_type = CE.chord_type_greedy(gt_roots, gt_notes_labels)
        pred_chord_name, pred_chord_type = CE.chord_type_greedy(pred_roots, n_pred)
        all_pred_notes_greedy.extend(pred_chord_name)

        pred_roots_, pred_names, pred_types = CE.chord_type_all(r_pred, n_pred)
        all_pred_roots.extend(CE.root_name(pred_roots_))
        all_pred_notes.extend(pred_names)


        all_gt_roots.extend(CE.root_name(gt_roots))
        all_gt_notes.extend(gt_chord_name)
        all_gt_beats.extend(data['beat_labels'].tolist())

    if modality == 'average_latent_features':
        song_roots = np.mean(song_roots, axis=0).reshape((1, song_roots.shape[1], song_roots.shape[2]))
        song_notes = np.mean(song_notes, axis=0).reshape((1, song_notes.shape[1], song_notes.shape[2]))
        song_beats = np.mean(song_beats, axis=0).reshape((1, song_beats.shape[1], song_beats.shape[2]))

        pred_beats = CE.beat_predictions(song_beats)
        average_pred_beats.extend(pred_beats.tolist())

        pred_roots = CE.root_predictions(song_roots)
        average_pred_roots_greedy.extend(CE.root_name(pred_roots))

        pred_chord_name, pred_chord_type = CE.chord_type_greedy(pred_roots, song_notes)
        average_pred_notes_greedy.extend(pred_chord_name)

        pred_roots_, pred_names, pred_types = CE.chord_type_all(song_roots, song_notes)
        average_pred_roots.extend(CE.root_name(pred_roots_))
        average_pred_notes.extend(pred_names)

        single_gt_roots.extend(CE.root_name(gt_roots))
        single_gt_notes.extend(gt_chord_name)
        single_gt_beats.extend(data['beat_labels'].tolist())

if modality == 'average_latent_features':
    total_root_greedy_acc = accuracy_score(single_gt_roots, average_pred_roots_greedy)
    total_root_acc = accuracy_score(single_gt_roots, average_pred_roots)

    total_notes_greedy_acc = accuracy_score(single_gt_notes, average_pred_notes_greedy)
    total_notes_acc = accuracy_score(single_gt_notes, average_pred_notes)

    total_beats_acc = accuracy_score(single_gt_beats, average_pred_beats)
    total_beats_precision = precision_score(single_gt_beats, average_pred_beats)
    total_beats_recall = recall_score(single_gt_beats, average_pred_beats)
else:
    total_root_greedy_acc = accuracy_score(all_gt_roots, all_pred_roots_greedy)
    total_root_acc = accuracy_score(all_gt_roots, all_pred_roots)

    total_notes_greedy_acc = accuracy_score(all_gt_notes, all_pred_notes_greedy)
    total_notes_acc = accuracy_score(all_gt_notes, all_pred_notes)

    total_beats_acc = accuracy_score(all_gt_beats, all_pred_beats)
    total_beats_precision = precision_score(all_gt_beats, all_pred_beats)
    total_beats_recall = recall_score(all_gt_beats, all_pred_beats)


results_name = model_name[:-3] + '_' + modality + '_results.txt'

with open(results_path + results_name  , 'w') as f:
    f.write('\nFor Model ' + results_name + '\n')
    f.write('    Roots_greedy' + '\n')
    f.write('        accuracy: {0:.2f}'.format(total_root_greedy_acc) + '\n')
    f.write('    Roots' + '\n')
    f.write('        accuracy: {0:.2f}'.format(total_root_acc) + '\n')
    f.write('    Notes_greedy' + '\n')
    f.write('        accuracy: {0:.2f}'.format(total_notes_greedy_acc) + '\n')
    f.write('    Notes' + '\n')
    f.write('        accuracy: {0:.2f}'.format(total_notes_acc) + '\n')
    f.write('    Beats' + '\n')
    f.write('        accuracy: {0:.2f}'.format(total_beats_acc) + '\n')
    f.write('        precision: {0:.2f}'.format(total_beats_precision) + '\n')
    f.write('        recall: {0:.2f}'.format(total_beats_recall) + '\n')

with open(results_path + results_name , 'r') as f:
    for line in f:
        print(line)


if modality == 'average_latent_features':
    matrix_name = model_name[:-3] + '_' + modality + '_roots_conf.eps'

    df_roots = CE.get_conf_matrix(single_gt_roots, average_pred_roots, labels=CE.roots)
    plt.figure(figsize = (10,7))
    # sn.set(font_scale=0.8)#for label size
    sn.heatmap(df_roots, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.savefig(results_path + matrix_name, format='eps', dpi=50)

    plt.clf()
    matrix_name = model_name[:-3] + '_' + modality + '_notes_conf.eps'

    df_notes = CE.get_conf_matrix(single_gt_notes, average_pred_notes, labels=CE.chord_types)
    plt.figure(figsize = (10,7))
    # sn.set(font_scale=0.8)#for label size
    sn.heatmap(df_notes, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.savefig(results_path + matrix_name, format='eps', dpi=50)
else:
    matrix_name = model_name[:-3] + '_' + modality + '_roots_conf.eps'

    df_roots = CE.get_conf_matrix(all_gt_roots, all_pred_roots, labels=CE.roots)
    plt.figure(figsize = (10,7))
    # sn.set(font_scale=0.8)#for label size
    sn.heatmap(df_roots, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.savefig(results_path + matrix_name, format='eps', dpi=50)

    plt.clf()
    matrix_name = model_name[:-3] + '_' + modality + '_notes_conf.eps'

    df_notes = CE.get_conf_matrix(all_gt_notes, all_pred_notes, labels=CE.chord_types)
    plt.figure(figsize = (10,7))
    # sn.set(font_scale=0.8)#for label size
    sn.heatmap(df_notes, cmap="Blues", annot=True,annot_kws={"size": 10}, fmt='g')
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.savefig(results_path + matrix_name, format='eps', dpi=50)



    # for gt, pred in zip(gt_root_names, pred_root_names):
    #     print('GT: {}, pred: {}'.format(gt, pred))

    # plt.plot(data['beat_labels'], 'x')
    # plt.plot(np.multiply(pred_beats, 2), 'x')
    # plt.show()
