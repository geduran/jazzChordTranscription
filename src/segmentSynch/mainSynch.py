#from midiData import MidiData, JsonData, InfoData
#from dataManager import BassOnsetDetector
from synchronizationManager import SynchronizationData
import scipy.io.wavfile
import glob
import os
import threading
import pickle
import numpy    as np


sr = 44100

# metric = 'cosine'
# metric = 'mahalanobis'
#metric = 'minkowski'
# metric = 'euclidean'

# bassOD = BassOnsetDetector()
results_dir = '../../results/segmentSynch/MIDI_results/'
data_dir = '../../data/MIDI/segmentSynch/'

prefix = '_noDrumsCqtRanges'



def synchDB_withMetric(metric):

    SM = SynchronizationData(sr=sr, workin_dir=results_dir)

    for dbName in ['0DB','1DB','2DB', '3DB']:
        print('\nStarting thread for db {} and metric {}! \n'.format(dbName, metric))

        # print('In Data Base ' + dbName + ' and metric ' + metric)
        files_dir = data_dir + dbName + '/'
        # dbName = workin_dir.split('/')[-2]

        files = glob.glob(files_dir + '*_noDrums.wav')

        db_error = {}
        all_hop_len = [2048, 4096, 8192]
        # for win_len in all_win_len:
        for hop_len in all_hop_len:
            # db_error['stft_win'+str(win_len)+'_hop'+str(hop_len)] = {}



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




            # db_error['cqt_hop'+str(hop_len)] = {}
            # db_error['tonnetz_hop'+str(hop_len)] = {}
            # # db_error['stft_onsets'] = {}
            # # db_error['tonnetz_onsets'] = {}
            # # db_error['cqt_onsets'] = {}
            # db_error['cqt_HFC_hop'+str(hop_len)] = {}
            # db_error['cqt_MFCC_hop'+str(hop_len)] = {}
            # db_error['cqt_tonnetz_hop'+str(hop_len)] = {}
            #
            # db_error['cqt_tonnetz_MFCC_hop'+str(hop_len)] = {}
            # db_error['tonnetz_MFCC_hop'+str(hop_len)] = {}
            # db_error['MFCC_hop'+str(hop_len)] = {}


        for file in files:
            if 'Bernie' in file:
                continue
            song_name = file.split('/')[-1].split('.')[0]

            _, audio = scipy.io.wavfile.read(file)
            audio = audio.astype('float32')

            if len(audio) < len(audio.flatten()):
                audio = (audio[:, 0] + audio[:, 1]) / 2

            # cnn_onset = bass_OD.cnn_onset_detect(audio)
            #
            # print('cnn_onset: {}'.format(cnn_onset))
            # print('rnn_onset: {}'.format(rnn_onsets))

            file = file.split('_noDrums')[0]
            choruses, audio_choruses = SM.get_audio_choruses(audio, file)
            choruses['dbName'] = dbName

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
                # print('processing song {} with hop_len {}, metric {}                      '.format(song_name, hop_len, metric), end='\r')

                # stft_result = SM.stft_synch(choruses, audio_choruses,
                #                             win_len=win_len, hop_len=hop_len,
                #                             metric=metric)
                # db_error['stft_win'+str(win_len)+'_hop'+str(hop_len)][song_name] = stft_result


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




                # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                #                             metric=metric, feature_list=['cqt', 'HFC'])
                # db_error['cqt_HFC_hop'+str(hop_len)][song_name] = mixed_result
                #
                # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                #                             metric=metric, feature_list=['cqt', 'MFCC'])
                # db_error['cqt_MFCC_hop'+str(hop_len)][song_name] = mixed_result
                #
                # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                #                             metric=metric, feature_list=['cqt', 'tonnetz'])
                # db_error['cqt_tonnetz_hop'+str(hop_len)][song_name] = mixed_result
                #
                # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                #                             metric=metric, feature_list=['cqt', 'tonnetz', 'MFCC'])
                # db_error['cqt_tonnetz_MFCC_hop'+str(hop_len)][song_name] = mixed_result
                #
                # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                #                             metric=metric, feature_list=['tonnetz', 'MFCC'])
                # db_error['tonnetz_MFCC_hop'+str(hop_len)][song_name] = mixed_result
                #
                # mixed_result = SM.mixed_features_synch(choruses, audio_choruses, hop_len=hop_len,
                #                             metric=metric, feature_list=['MFCC'])
                # db_error['MFCC_hop'+str(hop_len)][song_name] = mixed_result
                #
                # cqt_result = SM.cqt_synch(choruses, audio_choruses, hop_len=hop_len,
                #                           metric=metric)
                # db_error['cqt_hop'+str(hop_len)][song_name] = cqt_result
                #
                # tonnetz_result = SM.tonnetz_synch(choruses, audio_choruses, hop_len=hop_len,
                #                              metric=metric)
                # db_error['tonnetz_hop'+str(hop_len)][song_name] = tonnetz_result


                print('    Proccesed song {} with hop_len {}, db {}, metric {}!'.format(song_name, hop_len, dbName, metric))

                with open(files_dir +dbName+'_'+metric+'_errors'+prefix+'.pkl', 'wb') as f:
                    pickle.dump(db_error, f)

    print('\nThread for metric {} is ready! \n'.format(metric))

# for metric in ['euclidean', 'mahalanobis_cov','mahalanobis', 'cosine']:
#     synchDB_withMetric(metric)

print('\n\n\n Prefix is: {}'.format(prefix))

if not os.path.isdir(results_dir + '/paths_beats/'):
    os.system('mkdir '+results_dir+'/paths_beats/')

for dbName in ['0DB','1DB','2DB', '3DB']:
    if not os.path.isdir(results_dir + '/paths_beats/'+dbName+'/'):
        os.system('mkdir '+results_dir+'/paths_beats/'+dbName+'/')

# #['mahalanobis', 'mahalanobis_cov', 'cosine', 'euclidean']:
thread1 = threading.Thread(target=synchDB_withMetric, args=('mahalanobis',))
thread1.start()
thread2 = threading.Thread(target=synchDB_withMetric, args=('euclidean',))
thread2.start()

thread1.join()
thread2.join()

thread3 = threading.Thread(target=synchDB_withMetric, args=('mahalanobis_cov',))
thread3.start()
thread4 = threading.Thread(target=synchDB_withMetric, args=('cosine',))
thread4.start()

thread3.join()
thread4.join()


if os.path.isdir(results_dir + '/paths_beats/'):
    os.system('mv '+results_dir+'/paths_beats/ '+results_dir+'/paths_beats'+prefix+'/')
#



#
