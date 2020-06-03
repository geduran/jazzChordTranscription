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
sys.path.insert(0, '../chordTranscription/')
sys.path.insert(0, '../segmentSyncronization/')

from transcriptionManager import transcriptionData, leadSheetClass
from dlUtils import chordEval
from audioData import AudioData


"""
    Json file format should be:

    {
        "name": <song_name>,
        "boundaries": [{"start": start1, "end": end1},
                       {"start": start2, "end": end2}, ...]
        "num_measures": <num_measures>,
        "metre": "<num> / <den>"
    }

"""


if '-h' in sys.argv[1] or '--help' in sys.argv[1]:
    print('Programm that obtains a lead sheet from an audio.\n' +
          'The only input this programm accepts is the path to the json file'+
          ' containing the song name, segments boundaries and number of beats.')
    sys.exit(0)




audio_path = '../../data/JAAH/audios/wav/'
json_path = '../../data/JAAH/data/annotations_/'
leadSheetPath = '../../reports/chordTranscription/'

sr=44100
hop_len=2048

TM = transcriptionData(sr=sr, hop_len=hop_len)

config_path = sys.argv[1]


# Load config
config = TM.read_json(config_path)
song_name = config['name']
boundaries = config['boundaries']
num_measures = config['num_measures']
metre = config['metre']

LS = leadSheetClass(num_measures, metre)

if os.path.isfile(json_path + song_name + '_.json'):
    # We get the audio
    _, audio = AudioData.open_audio(audio_path + song_name + '.wav')

    # Get ground truth chord sequence
    gt_info = TM.get_json_info(json_path, song_name)
    choruses, _ = TM.get_audio_JAAH_choruses(audio,json_path,song_name)
    chords_gt = TM.getChords(choruses)

    LS.populateChords(chords_gt, 'gt')
else:
    # If is not a song from JAAH dataset, the wav file can be placed in this folder
    _, audio = AudioData.open_audio(song_name + '.wav')

# Chorus Synchronization
choruses = TM.segmentate_choruses(audio, boundaries)

print('Synchronizing choruses of ' + song_name)
dtw_paths = TM.synchronize_best(choruses, hop_len=hop_len)

cqt_choruses = TM.getCqtChoruses(choruses, hop_len=hop_len)#, info=gt_info)

warped_cqts = TM.warpCqts(cqt_choruses, dtw_paths)

# Chord prediction
averaged_pred, averaged_latent = TM.predict_chords(warped_cqts)

LS.populateChords(averaged_latent, 'pred')
# LS.populateChords(averaged_pred, 'predLatent')


# print('len gt: {}, len pred: {} '.format(len(aligned_chords), len(averaged_pred)))

LS.print(leadSheetPath + song_name)







#
