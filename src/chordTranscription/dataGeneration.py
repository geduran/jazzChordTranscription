import random
import pickle
import glob
import json
import time
import sys
import os
import numpy    as np
import pandas   as pd


# Own Code Imports
sys.path.insert(0, '../')
from transcriptionManager import transcriptionData
from audioData import AudioData

# from dlUtils import chordEval

audio_path = '../../data/JAAH/audios/wav_da/'
json_path = '../../data/JAAH/data/annotations_/'
alignments_path = '../../results/segmentSynch/JAAH_results/paths_beats/'
output_path = '../../data/JAAH/chordTranscription/both_da/'


not_useful = ['stompin_at_the_savoy',
              'st_louis_blues',
              'my_favorite_things',
              'watermelon_man',
              'django',
              'concerto_for_cootie',
              'potato_head_blues',
              'isfahan',
              'body_and_soul(hawkins)',
              'black_water_blues',
              'east_st_louis',
              'embraceable_you',
              'diminuendo_and_crescendo_in_blue',
              'indiana',
              'livery_stable_blues',
              'harlem_congo',
              'everybody_loves_my_baby',
              'dippermouth_blues',
              'grandpas_spells',
              'manteca',
              'from_monday_on',
              'the_stampede',
              'i_gotta_right_to_sing_the_blues',
              'these_foolish_things',
              'king_porter_stomp',
              'wrappin_it_up',
              'black_bottom_stomp',
              'shaw_nuff',
              'maple_leaf_rag(bechet)',
              'rockin_chair',
              'maple_leaf_rag(braxton)',
              'maple_leaf_rag(hyman)',
              'i_cant_get_started']

test_songs = [
              'breakfast_feud',
              'hotter_than_that',
              'body_and_soul(goodman)',
              'four_brothers',
              'minor_swing',
              'when_lights_are_low'
             ]


hop_len = 2048
sr = 44100
method = 'cqt'

# CE = chordEval()
TM = transcriptionData(sr=sr, hop_len=hop_len, workin_dir='',
                       dtw_results_path=alignments_path)
files = glob.glob(audio_path + '*wav')

total_time = 0
single_song_time = 0

for file in files:
    song_name= file.split('/')[-1].split('.')[0]

    if song_name[-3] == '_':
        i = int(song_name[-2:])
        song_name = song_name[:-3]
    else:
        i = int(song_name[-1])
        song_name = song_name[:-2]

    if song_name in not_useful:
        continue

    print('   Extracting data from song: ' + song_name +' shifting ' + str(i))

    _, audio = AudioData.open_audio(file)

    choruses, audio_choruses = TM.get_audio_JAAH_choruses(audio, json_path,
                                                          song_name)

    song_info = TM.get_json_info(json_path, song_name)

    cqt_choruses = TM.getCqtChoruses(audio_choruses, info=song_info, hop_len=hop_len, method=method)

    dtw_paths = TM.getDtwPaths(song_name)

    warped_cqts = TM.warpCqts(cqt_choruses, dtw_paths)

    averaged_cqt = TM.averageCqts(warped_cqts)

    chords = TM.getChords(choruses)

    aligned_chords, beat_labels = TM.alignChordsBeats(chords, choruses['0'], song_info,
                                         sr=sr, hop_len=hop_len)

    assert len(aligned_chords) == warped_cqts[0].shape[1]
    assert len(aligned_chords) == len(beat_labels)

    # print('    Has {} sequences and {} frames'.format(len(warped_cqts), len(aligned_chords)))

    coded_roots, coded_notes = TM.codeChordLabels(aligned_chords, transpose=i)



    #
    #
    # from dlUtils import chordEval, encode_labels
    # CE= chordEval()
    # name_root = CE.root_name(CE.root_predictions(coded_roots))
    # name_chord, _ = CE.chord_type_greedy(CE.root_predictions(coded_roots), coded_notes)
    #
    # chords = CE.to_chord_label(encode_labels(name_root, name_chord))
    #
    # for root, notes, chord in zip(name_root,  name_chord, chords):
    #     if chord !=  root +':'+ notes:
    #         print('chord {} -> {}:{}'.format(chord, root, notes))
    #
    # continue
    #
    #
    #









    # print('For i: {}, roots are: {} and notes are: {}'.format(i, coded_roots[:,50:55], coded_notes[:,50:55]))

    data = {
            'data': warped_cqts,
            'averaged_data': averaged_cqt,
            'root_labels': coded_roots,
            'notes_labels': coded_notes,
            'beats_labels': beat_labels,
            'hop_len': hop_len,
            'sr': sr,
            'method': method
            }

    # gt_roots = CE.root_predictions(coded_roots)
    # root_name = CE.root_name(gt_roots)
    # names, types = CE.chord_type_greedy(gt_roots, coded_notes)
    #
    # for chord, root, name in zip(aligned_chords, root_name, names):
    #     if chord != root+':'+name:
    #         print('chord: {} -> {}:{}'.format(chord, root, name))

    file_name = song_name +str(i)+ '_' + method + '_' + str(hop_len) + '.pkl'

    if song_name in test_songs:
        save_path = output_path + method + '/test/'
    else:
        save_path = output_path + method + '/'


    out_file = open(save_path + file_name, 'wb')
    pickle.dump(data, out_file)
    out_file.close()


    for _,value in choruses.items():
        if not isinstance(value, dict):
            continue
        single_song_time += value['end'] - value['start']
        break

    for _,value in choruses.items():
        if not isinstance(value, dict):
            continue
        total_time += value['end'] - value['start']


print('total_time: {0:0.2f}'.format(total_time/60))
print('single_song_time: {0:0.2f}'.format(single_song_time/60))

















#
