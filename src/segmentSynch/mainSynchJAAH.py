#from midiData import MidiData, JsonData, InfoData
#from dataManager import BassOnsetDetector
from synchronizationManager import SynchronizationData
import scipy.io.wavfile
import glob
import threading
import os
import sys
import pickle
import numpy    as np


sr = 44100

audio_path = '../../data/JAAH/audios/wav/'
json_path = '../../data/JAAH/data/annotations_/'
workin_dir = '../../results/segmentSynch/JAAH_results/'

prefix = '_cqtRanges'

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


#SM = SynchronizationData(sr=sr)
#bassOD = BassOnsetDetector()


def synchDB_withMetric(metric):
    print('\nStarting thread for {}! \n'.format(metric))

    SM = SynchronizationData(sr=sr, workin_dir=workin_dir)
    # bassOD = BassOnsetDetector()
    #print('\nDistance Function ' + metric, end='\n')

    files = glob.glob(audio_path + '*wav')


    db_error = {}
    all_hop_len = [2048]#, 4096, 8192]
    # for win_len in all_win_len:
    for hop_len in all_hop_len:
        # db_error['stft_win'+str(win_len)+'_hop'+str(hop_len)] = {}
        # db_error['stft_onsets'] = {}
        # db_error['tonnetz_onsets'] = {}
        # db_error['cqt_onsets'] = {}

        db_error['cqt_hop'+str(hop_len)] = {}
        db_error['tonnetz_hop'+str(hop_len)] = {}
        db_error['MFCC_hop'+str(hop_len)] = {}
        db_error['cqt_HFC_hop'+str(hop_len)] = {}
        db_error['cqt_MFCC_hop'+str(hop_len)] = {}
        db_error['cqt_tonnetz_hop'+str(hop_len)] = {}
        db_error['cqt_tonnetz_MFCC_hop'+str(hop_len)] = {}
        db_error['tonnetz_MFCC_hop'+str(hop_len)] = {}
        #

        db_error['cqt[ranges]_hop'+str(hop_len)] = {}
        db_error['cqt[ranges]_HFC_hop'+str(hop_len)] = {}
        db_error['cqt[ranges]_MFCC_hop'+str(hop_len)] = {}
        db_error['cqt[ranges]_tonnetz_hop'+str(hop_len)] = {}
        db_error['cqt[ranges]_tonnetz_MFCC_hop'+str(hop_len)] = {}

        db_error['cqt[bass]_cqt[mid]_hop'+str(hop_len)] = {}
        db_error['cqt[bass]_cqt[mid]_HFC_hop'+str(hop_len)] = {}
        db_error['cqt[bass]_cqt[mid]_MFCC_hop'+str(hop_len)] = {}
        db_error['cqt[bass]_cqt[mid]_tonnetz_hop'+str(hop_len)] = {}
        db_error['cqt[bass]_cqt[mid]_tonnetz_MFCC_hop'+str(hop_len)] = {}

        db_error['cqt[bass]_hop'+str(hop_len)] = {}
        db_error['cqt[bass]_HFC_hop'+str(hop_len)] = {}
        db_error['cqt[bass]_MFCC_hop'+str(hop_len)] = {}
        db_error['cqt[bass]_tonnetz_hop'+str(hop_len)] = {}
        db_error['cqt[bass]_tonnetz_MFCC_hop'+str(hop_len)] = {}

        db_error['cqt[mid]_hop'+str(hop_len)] = {}
        db_error['cqt[mid]_HFC_hop'+str(hop_len)] = {}
        db_error['cqt[mid]_MFCC_hop'+str(hop_len)] = {}
        db_error['cqt[mid]_tonnetz_hop'+str(hop_len)] = {}
        db_error['cqt[mid]_tonnetz_MFCC_hop'+str(hop_len)] = {}

        db_error['tonnetz[ranges]_hop'+str(hop_len)] = {}
        db_error['cqt_tonnetz[ranges]_hop'+str(hop_len)] = {}
        db_error['cqt_tonnetz[ranges]_MFCC_hop'+str(hop_len)] = {}
        db_error['tonnetz[ranges]_MFCC_hop'+str(hop_len)] = {}

        db_error['tonnetz[bass]_tonnetz[mid]_hop'+str(hop_len)] = {}
        db_error['cqt_tonnetz[bass]_tonnetz[mid]_hop'+str(hop_len)] = {}
        db_error['cqt_tonnetz[bass]_tonnetz[mid]_MFCC_hop'+str(hop_len)] = {}
        db_error['tonnetz[bass]_tonnetz[mid]_MFCC_hop'+str(hop_len)] = {}

        db_error['tonnetz[bass]_hop'+str(hop_len)] = {}
        db_error['cqt_tonnetz[bass]_hop'+str(hop_len)] = {}
        db_error['cqt_tonnetz[bass]_MFCC_hop'+str(hop_len)] = {}
        db_error['tonnetz[bass]_MFCC_hop'+str(hop_len)] = {}

        db_error['tonnetz[mid]_hop'+str(hop_len)] = {}
        db_error['cqt_tonnetz[mid]_hop'+str(hop_len)] = {}
        db_error['cqt_tonnetz[mid]_MFCC_hop'+str(hop_len)] = {}
        db_error['tonnetz[mid]_MFCC_hop'+str(hop_len)] = {}


        # db_error['cqt_bass[tonnetz]_hop'+str(hop_len)] = {}
        # db_error['tonnetz_bass[tonnetz]_hop'+str(hop_len)] = {}
        # db_error['MFCC_bass[tonnetz]_hop'+str(hop_len)] = {}
        # db_error['cqt_HFC_bass[tonnetz]_hop'+str(hop_len)] = {}
        # db_error['cqt_MFCC_bass[tonnetz]_hop'+str(hop_len)] = {}
        # db_error['cqt_tonnetz_bass[tonnetz]_hop'+str(hop_len)] = {}
        # db_error['cqt_tonnetz_MFCC_bass[tonnetz]_hop'+str(hop_len)] = {}
        # db_error['tonnetz_MFCC_bass[tonnetz]_hop'+str(hop_len)] = {}
        #
        # db_error['cqt_bass[cqt]_hop'+str(hop_len)] = {}
        # db_error['tonnetz_bass[cqt]_hop'+str(hop_len)] = {}
        # db_error['MFCC_bass[cqt]_hop'+str(hop_len)] = {}
        # db_error['cqt_HFC_bass[cqt]_hop'+str(hop_len)] = {}
        # db_error['cqt_MFCC_bass[cqt]_hop'+str(hop_len)] = {}
        # db_error['cqt_tonnetz_bass[cqt]_hop'+str(hop_len)] = {}
        # db_error['cqt_tonnetz_MFCC_bass[cqt]_hop'+str(hop_len)] = {}
        # db_error['tonnetz_MFCC_bass[cqt]_hop'+str(hop_len)] = {}

    for file in files:
        song_name = file.split('/')[-1].split('.')[0]

        if song_name in not_useful:
            continue

        if 'dancers' not in song_name:
            continue

######################################################################################################
        # if 'giant_steps' not in song_name:
        #     continue
######################################################################################################
        print('Analyzing song: ' + song_name)

        _, audio = scipy.io.wavfile.read(file)
        audio = audio.astype('float32')

        if len(audio) < len(audio.flatten()):
            audio = (audio[:, 0] + audio[:, 1]) / 2

        # cnn_onset = bass_OD.cnn_onset_detect(audio)
        #
        # print('cnn_onset: {}'.format(cnn_onset))
        # print('rnn_onset: {}'.format(rnn_onsets))

        choruses, audio_choruses = SM.get_audio_JAAH_choruses(audio, json_path, song_name)

        #print('song: {}, choruses: {}, audio_choruses: {}'.format(song_name, len(choruses), len(audio_choruses)))
        # rnn_onsets = bassOD.rnn_onset_detect(audio_choruses)
        # ###
        # # for i, audio_seg in enumerate(audio_choruses):
        #     # audio_data = bassOD.get_rnn_audio_data(audio_seg)
        #     # bassOD.check_segmentation(audio_data, song_name + str(i)+'_rnn', rnn_onsets[i])
        # ###
        # rnn_stft = SM.custom_synch(choruses, audio_choruses, rnn_onsets,
        #                              metric=metric, method='stft')
        #
        # db_error['stft_onsets'][song_name] = rnn_stft
        #
        # rnn_cqt = SM.custom_synch(choruses, audio_choruses, rnn_onsets,
        #                              metric=metric, method='cqt')
        #
        # db_error['cqt_onsets'][song_name] = rnn_cqt
        #
        # rnn_tonnetz = SM.custom_synch(choruses, audio_choruses, rnn_onsets,
        #                              metric=metric, method='tonnetz')
        #
        # db_error['tonnetz_onsets'][song_name] = rnn_tonnetz


        # for win_len in all_win_len:
        for hop_len in all_hop_len:
            #print('processing song {} with hop_len {}, metric {}                      '.format(song_name, hop_len, metric), end='\r')

            # stft_result = SM.stft_synch(choruses, audio_choruses,
            #                             win_len=win_len, hop_len=hop_len,
            #                             metric=metric)
            # db_error['stft_win'+str(win_len)+'_hop'+str(hop_len)][song_name] = stft_result

################################################################################################
            if False:

                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'HFC'])
                db_error['cqt_HFC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'MFCC'])
                db_error['cqt_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'tonnetz'])
                db_error['cqt_tonnetz_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'tonnetz', 'MFCC'])
                db_error['cqt_tonnetz_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['tonnetz', 'MFCC'])
                db_error['tonnetz_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['MFCC'])
                db_error['MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt'])
                db_error['cqt_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['tonnetz'])
                db_error['tonnetz_hop'+str(hop_len)][song_name] = mixed_result



################################################################################################

            if True:
                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[ranges]', 'HFC'])
                db_error['cqt[ranges]_HFC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[ranges]', 'MFCC'])
                db_error['cqt[ranges]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[ranges]', 'tonnetz'])
                db_error['cqt[ranges]_tonnetz_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[ranges]', 'tonnetz', 'MFCC'])
                db_error['cqt[ranges]_tonnetz_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[ranges]'])
                db_error['cqt[ranges]_hop'+str(hop_len)][song_name] = mixed_result


################################################################################################
            if False:

                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[bass]', 'HFC'])
                db_error['cqt[bass]_HFC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[bass]', 'MFCC'])
                db_error['cqt[bass]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[bass]', 'tonnetz'])
                db_error['cqt[bass]_tonnetz_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[bass]', 'tonnetz', 'MFCC'])
                db_error['cqt[bass]_tonnetz_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[bass]'])
                db_error['cqt[bass]_hop'+str(hop_len)][song_name] = mixed_result

################################################################################################

            if False:
                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[mid]', 'HFC'])
                db_error['cqt[mid]_HFC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[mid]', 'MFCC'])
                db_error['cqt[mid]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[mid]', 'tonnetz'])
                db_error['cqt[mid]_tonnetz_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[mid]', 'tonnetz', 'MFCC'])
                db_error['cqt[mid]_tonnetz_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[mid]'])
                db_error['cqt[mid]_hop'+str(hop_len)][song_name] = mixed_result

################################################################################################

            if False:
                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[bass]', 'cqt[mid]', 'HFC'])
                db_error['cqt[bass]_cqt[mid]_HFC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[bass]', 'cqt[mid]', 'MFCC'])
                db_error['cqt[bass]_cqt[mid]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[bass]', 'cqt[mid]', 'tonnetz'])
                db_error['cqt[bass]_cqt[mid]_tonnetz_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[bass]', 'cqt[mid]', 'tonnetz', 'MFCC'])
                db_error['cqt[bass]_cqt[mid]_tonnetz_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt[bass]', 'cqt[mid]'])
                db_error['cqt[bass]_cqt[mid]_hop'+str(hop_len)][song_name] = mixed_result



################################################################################################

            if False:
                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'tonnetz[ranges]'])
                db_error['cqt_tonnetz[ranges]_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'tonnetz[ranges]', 'MFCC'])
                db_error['cqt_tonnetz[ranges]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['tonnetz[ranges]', 'MFCC'])
                db_error['tonnetz[ranges]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['tonnetz[ranges]'])
                db_error['tonnetz[ranges]_hop'+str(hop_len)][song_name] = mixed_result

################################################################################################
            if False:
                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'tonnetz[bass]'])
                db_error['cqt_tonnetz[bass]_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'tonnetz[bass]', 'MFCC'])
                db_error['cqt_tonnetz[bass]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['tonnetz[bass]', 'MFCC'])
                db_error['tonnetz[bass]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['tonnetz[bass]'])
                db_error['tonnetz[bass]_hop'+str(hop_len)][song_name] = mixed_result

################################################################################################

            if False:
                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'tonnetz[mid]'])
                db_error['cqt_tonnetz[mid]_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'tonnetz[mid]', 'MFCC'])
                db_error['cqt_tonnetz[mid]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['tonnetz[mid]', 'MFCC'])
                db_error['tonnetz[mid]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['tonnetz[mid]'])
                db_error['tonnetz[mid]_hop'+str(hop_len)][song_name] = mixed_result

################################################################################################

            if False:
                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'tonnetz[bass]', 'tonnetz[mid]'])
                db_error['cqt_tonnetz[bass]_tonnetz[mid]_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['cqt', 'tonnetz[bass]', 'tonnetz[mid]', 'MFCC'])
                db_error['cqt_tonnetz[bass]_tonnetz[mid]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['tonnetz[bass]', 'tonnetz[mid]', 'MFCC'])
                db_error['tonnetz[bass]_tonnetz[mid]_MFCC_hop'+str(hop_len)][song_name] = mixed_result


                mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                                            metric=metric, feature_list=['tonnetz[bass]', 'tonnetz[mid]'])
                db_error['tonnetz[bass]_tonnetz[mid]_hop'+str(hop_len)][song_name] = mixed_result
#####################################################################################################

# Only bass range cqt
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['cqt', 'HFC','tonnetz[bass]'])
            # db_error['cqt_HFC_tonnetz[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_HFC_bass[]_hop'+str(hop_len) +' is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['cqt', 'MFCC','tonnetz[bass]'])
            # db_error['cqt_MFCC_tonnetz[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_MFCC_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['cqt', 'tonnetz','tonnetz[bass]'])
            # db_error['cqt_tonnetz_tonnetz[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_tonnetz_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['cqt', 'tonnetz', 'MFCC','tonnetz[bass]'])
            # db_error['cqt_tonnetz_MFCC_tonnetz[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_tonnetz_MFCC_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['tonnetz', 'MFCC','tonnetz[bass]'])
            # db_error['tonnetz_MFCC_tonnetz[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with tonnetz_MFCC_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['MFCC','tonnetz[bass]'])
            # db_error['MFCC_tonnetz[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with MFCC_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['cqt','tonnetz[bass]'])
            # db_error['cqt_tonnetz[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_tonnetz_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['tonnetz','tonnetz[bass]'])
            # db_error['tonnetz_tonnetz[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_tonnetz_bass[]_hop is {}'.format(mixed_result))

# #####################################################################################################


# Only bass range tonnetz
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['cqt', 'HFC', 'cqt[bass]'])
            # db_error['cqt_HFC_cqt[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_HFC_bass[]_hop'+str(hop_len) +' is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['cqt', 'MFCC', 'cqt[bass]'])
            # db_error['cqt_MFCC_cqt[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_MFCC_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['cqt', 'tonnetz', 'cqt[bass]'])
            # db_error['cqt_tonnetz_cqt[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_tonnetz_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['cqt', 'tonnetz', 'MFCC', 'cqt[bass]'])
            # db_error['cqt_tonnetz_MFCC_cqt[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_tonnetz_MFCC_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['tonnetz', 'MFCC', 'cqt[bass]'])
            # db_error['tonnetz_MFCC_cqt[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with tonnetz_MFCC_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['MFCC', 'cqt[bass]'])
            # db_error['MFCC_cqt[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with MFCC_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['cqt', 'cqt[bass]'])
            # db_error['cqt_cqt[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_tonnetz_bass[]_hop is {}'.format(mixed_result))
            # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
            #                             metric=metric, feature_list=['tonnetz', 'cqt[bass]'])
            # db_error['tonnetz_cqt[bass]_hop'+str(hop_len)][song_name] = mixed_result
            # #print('Curr error with cqt_tonnetz_bass[]_hop is {}'.format(mixed_result))


            print('    Ready song {} with hop_len {} and metric {} !'.format(song_name, hop_len, metric))

            #with open(workin_dir +metric+'_errors'+prefix+'.pkl', 'wb') as f:
             #  pickle.dump(db_error, f)
#
#     print('\nThread for metric {} is ready! \n'.format(metric))
#
# if not os.path.isdir(workin_dir + '/paths_beats/'):
#     os.system('mkdir '+workin_dir+'/paths_beats/')
#
# print('\n\n\n Prefix is: {}'.format(prefix))
# # ['mahalanobis', 'mahalanobis_cov', 'cosine', 'euclidean']:
# thread1 = threading.Thread(target=synchDB_withMetric, args=('mahalanobis_cov',))
# thread1.start()
# thread2 = threading.Thread(target=synchDB_withMetric, args=('euclidean',))
# thread2.start()
#
# thread1.join()
# thread2.join()

thread3 = threading.Thread(target=synchDB_withMetric, args=('mahalanobis',))
thread3.start()
# thread4 = threading.Thread(target=synchDB_withMetric, args=('cosine',))
# thread4.start()
#
thread3.join()
# thread4.join()
#
# if os.path.isdir(workin_dir + '/paths_beats/'):
#     os.system('mv '+workin_dir+'/paths_beats/ '+workin_dir+'/paths_beats'+prefix+'/')









#
