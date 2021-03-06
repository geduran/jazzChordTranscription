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



audio_path = '../../data/JAAH/audios/wav/'
json_path = '../../data/JAAH/data/annotations_/'
alignments_path = '../../results/segmentSynch/JAAH_results/paths_beats/'
output_path = '../../data/JAAH/chordTranscription/'


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

hop_len = 2048
sr = 44100
method = 'cqt'


TM = transcriptionData(sr=sr, hop_len=hop_len, workin_dir='',
                       dtw_results_path=alignments_path)
files = glob.glob(audio_path + '*wav')

total_time = 0
single_song_time = 0

for file in files:
    song_name = file.split('/')[-1].split('.')[0]

    if song_name in not_useful:
        continue

    # if 'hotter_than' not in song_name:
    #     continue

    print('Extracting data from song: ' + song_name)

    _, audio = AudioData.open_audio(file)
    choruses, audio_choruses = TM.get_audio_JAAH_choruses(audio, json_path,
                                                          song_name)

    song_info = TM.get_json_info(json_path, song_name)

    cqt_choruses = TM.getCqtChoruses(audio_choruses, song_info, hop_len=hop_len, method=method)

    dtw_paths = TM.getDtwPaths(song_name)

    warped_cqts = TM.warpCqts(cqt_choruses, dtw_paths)

    averaged_cqt = TM.averageCqts(warped_cqts)

    chords = TM.getChords(choruses)

    aligned_chords, beat_labels = TM.alignChordsBeats(chords, choruses['0'], song_info,
                                         sr=sr, hop_len=hop_len)

    assert len(aligned_chords) == warped_cqts[0].shape[1]
    assert len(aligned_chords) == len(beat_labels)

    print('    Has {} sequences and {} frames'.format(len(warped_cqts), len(aligned_chords)))

    coded_roots, coded_notes = TM.codeChordLabels(aligned_chords)


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

    file_name = song_name + '_' + method + '_' + str(hop_len) + '.pkl'
    save_path = output_path + method + '/'

    out_file = open(save_path + file_name, 'wb')
    pickle.dump(data, out_file)
    out_file.close()




    # for _,value in choruses.items():
    #     if not isinstance(value, dict):
    #         continue
    #     single_song_time += value['end'] - value['start']
    #     break
    #
    # for _,value in choruses.items():
    #     if not isinstance(value, dict):
    #         continue
    #     total_time += value['end'] - value['start']


# print('total_time: {0:0.2f}'.format(total_time/60))
# print('single_song_time: {0:0.2f}'.format(single_song_time/60))

















#
