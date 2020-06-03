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

from transcriptionManager import transcriptionData
from dlUtils import chordEval
from audioData import AudioData


audio_path = '../../data/JAAH/audios/wav/'
json_path = '../../data/JAAH/data/annotations_/'
sr=44100
hop_len=2048

TM = transcriptionData(sr=sr, hop_len=hop_len)
CE = chordEval()

# Load config
config = TM.read_json('example2.json')
song_name = config['name']
boundaries = config['boundaries']
num_measures = config['num_measures']
metre = config['metre']

# We get the audio
_, audio = AudioData.open_audio(audio_path + song_name + '.wav')

# Get ground truth chord sequence
gt_info = TM.get_json_info(json_path, song_name)
choruses, _ = TM.get_audio_JAAH_choruses(audio,json_path,song_name)
chords_gt = TM.getChords(choruses)
aligned_chords, _ = TM.alignChordsBeats(chords_gt, choruses['0'],
                                                  gt_info, sr=sr,
                                                  hop_len=hop_len)

print('chord: {}'.format(aligned_chords[-230]))
coded_roots, coded_notes = TM.codeChordLabels(aligned_chords, transpose=0)

print('coded_root: {}'.format(coded_roots[:,-230]))
print('coded_notes: {}'.format(coded_notes[:,-230]))

name_root = CE.root_name(CE.root_predictions(coded_roots))
name_chord, _ = CE.chord_type_greedy(CE.root_predictions(coded_roots), coded_notes)

print('name_root: {}'.format(name_root[-230]))
print('name_chord: {}'.format(name_chord[-230]))


# for curr_chord, curr_root, curr_type in zip(aligned_chords, name_root, name_chord):
#     print('{} -> {}:{}'.format(curr_chord, curr_root, curr_type))






"""


"|Ab:maj6 Eb:7 |Ab:maj6 Eb:7 |Ab:maj6 Eb:7 |Ab:maj6 F:7|Bb:min7 |C:hdim7 F:7 |Bb:min7 Eb:7 |Ab:maj6 Eb:7 |",
"|Ab:maj6 Eb:7 |Ab:maj6 Eb:7 |Ab:maj6 Eb:7 |Ab:maj6 F:7|Bb:min7 |C:hdim7 F:7 |Bb:min7 Eb:7 |Ab:maj6 |",
"|Db:min7 Gb:7 |B:maj7 |E:min7 A:min7 |D:maj7 |G:min7 C:7 |F:maj7 |F:min7 Bb:7 |Bb:min7 Eb:(3,5,b7,b9) |",
"|Ab:maj6 Eb:7 |Ab:maj6 Eb:7 |Ab:maj6 Eb:7 |Ab:maj6 F:7|Bb:min7 |C:hdim7 F:7 |Bb:min7 Eb:7 |Ab:maj6 |"



"""








#
