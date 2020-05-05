
# from midiData import MidiData, JsonData, InfoData
# from dataManager import BassOnsetDetector
from synchronizationManager import get_average_results, get_comparations
import scipy.io.wavfile
import glob
import os
import pickle
import numpy    as np


sr = 44100
workin_dir = '../../results/segmentSynch/MIDI_results/'

metrics = ['euclidean', 'mahalanobis','mahalanobis_cov',
           'cosine']
# metrics = [ 'mahalanobis', 'correlation', 'hamming', 'canberra']
dbNames = ['0DB','1DB','2DB','3DB']
# SM = SynchronizationData(sr=sr)
prefix = '_noDrumsCqtRanges'


feature_combinations = ['cqt',
                        'tonnetz',
                        'MFCC',
                        'cqt_HFC',
                        'cqt_MFCC',
                        'cqt_tonnetz',
                        'cqt_tonnetz_MFCC',
                        'tonnetz_MFCC',
                        'cqt[ranges]',
                        'cqt[ranges]_HFC',
                        'cqt[ranges]_MFCC',
                        'cqt[ranges]_tonnetz',
                        'cqt[ranges]_tonnetz_MFCC',

                        'cqt[bass]_cqt[mid]',
                        'cqt[bass]_cqt[mid]_HFC',
                        'cqt[bass]_cqt[mid]_MFCC',
                        'cqt[bass]_cqt[mid]_tonnetz',
                        'cqt[bass]_cqt[mid]_tonnetz_MFCC',

                        'cqt[bass]',
                        'cqt[bass]_HFC',
                        'cqt[bass]_MFCC',
                        'cqt[bass]_tonnetz',
                        'cqt[bass]_tonnetz_MFCC',

                        'cqt[mid]',
                        'cqt[mid]_HFC',
                        'cqt[mid]_MFCC',
                        'cqt[mid]_tonnetz',
                        'cqt[mid]_tonnetz_MFCC',

                        'tonnetz[ranges]',
                        'cqt_tonnetz[ranges]',
                        'cqt_tonnetz[ranges]_MFCC',
                        'tonnetz[ranges]_MFCC',

                        'tonnetz[bass]_tonnetz[mid]',
                        'cqt_tonnetz[bass]_tonnetz[mid]',
                        'cqt_tonnetz[bass]_tonnetz[mid]_MFCC',
                        'tonnetz[bass]_tonnetz[mid]_MFCC',

                        'tonnetz[bass]',
                        'cqt_tonnetz[bass]',
                        'cqt_tonnetz[bass]_MFCC',
                        'tonnetz[bass]_MFCC',

                        'tonnetz[mid]',
                        'cqt_tonnetz[mid]',
                        'cqt_tonnetz[mid]_MFCC',
                        'tonnetz[mid]_MFCC'
                        ]


for metric in metrics:
    for db in dbNames:
        prefix_ = prefix
        dbName_ = db
        if len(prefix) and '_' not in prefix[0]:
            prefix_ = '_' + prefix
        if len(db):
            dbName_ =  dbName_+'_'
        path = workin_dir + dbName_ + metric+ '_errors'+prefix_+'.pkl'
        print(path)
        print(' ')
        print(' ')
        if not os.path.isfile(path):
             print('File {} not available yet...'.format(metric+ '_errors'+prefix_))
             continue
        get_average_results(workin_dir, db, metric, prefix)
        get_comparations(workin_dir, db, metric, prefix, feature_combinations)
        print('\n')
