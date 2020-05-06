import madmom
import mido
import scipy.io.wavfile
import numpy as np
import librosa.display
import librosa
import matplotlib.pyplot as plt
from madmom.audio.filters import MelFilterbank
import random
import math
import os
import pickle
import glob
import json
import time
import sys
import numpy    as np
import pandas   as pd
sys.path.insert(0, '../')
from midiData import MidiData, JsonData, InfoData



class SynchronizationData():

    def __init__(self, sr=44100, workin_dir='.', hop_len=512):
        self.sr = sr
        self.hop_len = hop_len
        self.workin_dir = workin_dir

    def read_json(self, path):
        if '.json' not in path:
            path += '.json'
        file = open(path, 'r')
        data = json.load(file)
        file.close()
        return data

    def get_json_info(self, json_path, song_name):
        json_data = self.read_json(json_path + song_name + '_')
        if 'key' in json_data['sandbox'].keys():
            key = json_data['sandbox']['key']
        else:
            key = None
        info = {
            'tuning': json_data['tuning'],
            'metric': json_data['metre'],
            'key': key
        }
        return info

    def _get_JAAH_choruses(self, json_data):
        choruses = {}
        valid_matching_segments = json_data['matching_segments']
        part_cont = 0
        for valid_parts in valid_matching_segments:
            for part in valid_parts:
                curr_segment = json_data['parts'][part]
                beats = curr_segment['beats']
                chords = curr_segment['chords']
                end_beat = 2*beats[-1] - beats[-2]
                choruses[str(part_cont)] = {'beats': beats, 'start': beats[0],
                                       'end': end_beat, 'chords': chords}
                part_cont += 1
        return choruses

    def get_audio_JAAH_choruses(self, audio, json_path, song_name):
        """
            Segments the audioand the data by choruses. The info is returned in
            a dict called choruses (key indicates index and value the conten),
            and the audio in a list.

            :param audio: input audio
            :type audio: np.array
            :param json_path: input path to json file
            :type json_path: str
            :param song_name: name of the current song
            :type song_name: str
            :return: dict with info (beats, etc), list with audio choruses
        """

        json_data = self.read_json(json_path + song_name + '_')


        choruses = self._get_JAAH_choruses(json_data)
        choruses['name'] = song_name
        audio_choruses = self.segmentate_choruses(audio, choruses)

        return choruses, audio_choruses


    def get_audio_choruses(self, audio, fileName):
        info = InfoData(fileName)

        choruses = info.get_choruses()
        choruses['name'] = fileName.split('/')[-1]

        audio_choruses = self.segmentate_choruses(audio, choruses)

        return choruses, audio_choruses

    def segmentate_choruses(self, audio, choruses):
        audio_choruses = []
        for key, values in choruses.items():
            if 'ame' in key:
                continue
            curr_start = int(values['start'] * self.sr)
            curr_end = int(values['end'] * self.sr)
            audio_choruses.append(audio[curr_start:curr_end]/np.max(audio[curr_start:curr_end]))
            # scipy.io.wavfile.write('/Users/gabrielduran007/Desktop/University/MAGISTER/codigos/jazz_JAAH/' + choruses['name'] + '_seg'+key+'.wav', self.sr, audio[curr_start:curr_end]/np.max(audio[curr_start:curr_end]))
        return audio_choruses

    def curr_dist(self, p1, p2, p3):
        #print('p1: {}, p2: {}, p3: {}'.format(p1, p2, p3))
        """ segment line AB, point P, where each one is an array([x, y]) """
        if all(p1 == p3) or all(p2 == p3):
            return 0
        if np.arccos(np.dot((p3 - p1) / np.linalg.norm(p3 - p1), (p2 - p1) / np.linalg.norm(p2 - p1)) / np.pi) > np.pi / 2:
            return np.linalg.norm(p3 - p1)
        if np.arccos(np.dot((p3 - p2) / np.linalg.norm(p3 - p2), (p1 - p2) / np.linalg.norm(p1 - p2)) / np.pi) > np.pi / 2:
            return np.linalg.norm(p3 - p2)
        return np.linalg.norm(np.cross(p1-p2, p1-p3))/np.linalg.norm(p2-p1)


    def frame_error(self, beats1, beats2, frame):
        min_error = 1e10
        min_pos = 0
        # print('frame: {} beats1: {} beats2: {}'.format(frame, beats1, beats2))
        if frame[0] > beats1[-1] or frame[1] > beats2[-1]:
            index1 = len(beats1) - 1
            index2 = len(beats2) - 1
        else:
            index1 = np.min(np.argwhere(np.array(beats1) >= np.array(frame[0])))
            index2 = np.min(np.argwhere(np.array(beats2) >= np.array(frame[1])))
        min_index = max(min(index1, index2)-2, 0)
        max_index = min(max(index1, index2)+2, len(beats1)-1)

        for i in range(min_index, max_index):
            p1 = np.array([beats1[i], beats2[i]])
            p2 = np.array([beats1[i+1], beats2[i+1]])
            curr_error = self.curr_dist(p1, p2, frame)
            if curr_error != curr_error:
                break
            if curr_error < min_error:
                min_pos = [p1, p2, frame]
            min_error = min(curr_error, min_error)

        return min_error


    def path_error(self, seg1, seg2, path):
        #print('beats: {}, start: {}'.format(seg1['beats'], seg1['start']))

        beats1 = list(np.array(seg1['beats']) - seg1['start'])
        beats1.append(seg1['end'] - seg1['start'])
        beats2 = list(np.array(seg2['beats']) - seg2['start'])
        beats2.append(seg2['end'] - seg2['start'])

        total_error = 0
        for frame in path:
            #frame = frame * self.hop_len/self.sr
            total_error += abs(self.frame_error(beats1, beats2, frame))#**2

        total_error /= len(path)
        return total_error

    def plot_path_errors(self, seg1, seg2, path, dbName='', name='',
                         pair=(0,0), method='', metric='euclidean'):

        beats1 = list(np.array(seg1['beats']) - seg1['start'])
        beats1.append(seg1['end'] - seg1['start'])
        beats2 = list(np.array(seg2['beats']) - seg2['start'])
        beats2.append(seg2['end'] - seg2['start'])

        dir =  self.workin_dir + 'pathErrors/'

        plt.clf()
        plt.plot(beats1, beats2, 'b')
        plt.plot(path[:,0], path[:,1], 'r')

        subDir = dir + dbName + '/'

        if not os.path.isdir(subDir):
           os.system('mkdir ' + subDir)

        plotPath = subDir + name+'_'+str(pair[0])+'_'+str(pair[1]) +'_' +method+'_'+metric+'.eps'
        plt.savefig(plotPath, format='eps',  dpi=40)
        plt.clf()


        # save_info = {'beats1': beats1, 'beats2': beats2, 'dtw_path': path,
        #              'pair': pair}
        # plotPath = '../../segmentSyncronization/paths_beats/' + dbName + '/' + name+'_'+str(pair[0])+'_'+str(pair[1]) +'_' +method+'_'+metric+'.eps'
        # with open(plotPath[:-3] + 'pkl', 'wb') as file:
        #     pickle.dump(save_info, file)

    def save_path_beats(self, seg1, seg2, path, dbName='', name='', pair=(0,0),
                        method='', metric='', hop_len=''):

        beats1 = list(np.array(seg1['beats']) - seg1['start'])
        beats1.append(seg1['end'] - seg1['start'])
        beats2 = list(np.array(seg2['beats']) - seg2['start'])
        beats2.append(seg2['end'] - seg2['start'])
        save_info = {'beats1': beats1, 'beats2': beats2, 'dtw_path': path,
                     'pair': pair}
        plotPath = self.workin_dir + 'paths_beats/' + dbName + '/' + name+'_'+str(pair[0])+'_'+str(pair[1]) +'_' +method+'_'+hop_len+'_'+metric+'.pkl'
        with open(plotPath, 'wb') as file:
            pickle.dump(save_info, file)

    def two_seq_dtw(self, chroma1, chroma2, chorus1, chorus2, metric='euclidean', printTimes=True,
                    printError=True, dbName='',name='', pair=(0,0), method='', inv_cov=None, hop_len=''):

        chroma_value1 = chroma1['chroma_values']
        chroma_value2 = chroma2['chroma_values']

        index_seconds1 = chroma1['index_second']
        index_seconds2 = chroma2['index_second']

        t0 = time.time()
        # print('chroma1: {}, chroma2: {}'.format(chroma_value1.shape, chroma_value2.shape))
        cost_matrix, path_dtw = librosa.sequence.dtw(X=chroma_value1,
                                                     Y=chroma_value2,
                                                     metric=metric,
                                                     mahalanobis_inv_cov=inv_cov)

        ########################################################################
        # print('Saving data of pair {}'.format(pair))
        # file = open('../../notebooks/giant_steps_paths/features_giant_steps_'+str(pair[0])+'_'+str(pair[1])+'.pkl', 'wb')
        #
        # info = {'features_'+str(pair[0]): chroma_value1,
        #         'features_'+str(pair[1]): chroma_value2,
        #         'beats_'+str(pair[0]): list(np.array(chorus1['beats']) - chorus1['start']),
        #         'beats_'+str(pair[1]): list(np.array(chorus2['beats']) - chorus2['start']),
        #         'path': path_dtw}
        #
        # pickle.dump(info, file)
        # file.close()
        ########################################################################

        path_dtw = path_dtw[::-1]
        path_dtw_seconds = np.ones(path_dtw.shape)
        for i, frame in enumerate(path_dtw):
            path_dtw_seconds[i,0] = index_seconds1[frame[0]]
            path_dtw_seconds[i,1] = index_seconds2[frame[1]]

        t1 = time.time()
        if printTimes:
            print('dtw calculation took {0:.2f} seconds'.format(t1-t0))

        curr_error = self.path_error(chorus1, chorus2, path_dtw_seconds)


        self.save_path_beats(chorus1, chorus2, path_dtw_seconds, name=name,
                             dbName=dbName, pair=pair, method=method,
                             metric=metric, hop_len=hop_len)


        #self.plot_path_errors(chorus1, chorus2, path_dtw_seconds, name=name,
        #                      dbName=dbName, pair=pair, method=method, metric=metric)

        if printTimes:
            print('error calculation took {0:.2f} seconds'.format(time.time()-t1))

        if printError:
            print('Path MAE is {:.2e}\n'.format(curr_error))

        return path_dtw, curr_error

    def all_seq_dtw(self, chromas, choruses, metric='euclidean',
                    printTimes=True, printError=True, method='', hop_len=''):

        dbName = ''
        if 'dbName' in choruses.keys():
            dbName = choruses['dbName']

        t0 = time.time()
        if 'mahalanobis_cov' in metric:
            metric = 'mahalanobis'
            all_features = None
            for chroma_dict in chromas:
                curr_chroma = chroma_dict['chroma_values']
                if all_features is None:
                    all_features = curr_chroma
                else:
                    all_features = np.concatenate((all_features, curr_chroma), axis=1)
            cov = np.cov(all_features)
            inv_cov = np.linalg.inv(cov)
        else:
            inv_cov = None
        # chromas are list of dicts, containing chroma values and index times.
        cumulated_error = []
        error_matrix = np.zeros((len(chromas), len(chromas)))
        #path_matrix = {}
        #error_matrix = np.zeros((len(chromas), len(chromas2)))

        for i, chroma1 in enumerate(chromas):
            for j , chroma2 in enumerate(chromas):
                if i != j and j > i:
                    curr_path, curr_error = self.two_seq_dtw(chroma1, chroma2, choruses[str(i)],
                                                choruses[str(j)], metric=metric,
                                                printTimes=printTimes, printError=printError,
                                                dbName=dbName,
                                                name=choruses['name'],
                                                pair=(i,j), method=method, inv_cov=inv_cov, hop_len=hop_len)
                    cumulated_error.append(curr_error)
                    error_matrix[i,j] = curr_error
                    error_matrix[j,i] = curr_error
                    #path_matrix[str(i) + '_' + str(j)]  = curr_path
                    #path_matrix[str(j) + '_' + str(i)]  = curr_path

        mean_error = sum(cumulated_error) / len(cumulated_error)

        return { 'mean_error':   mean_error,
                 'error_matrix': error_matrix}#,
                # 'path_matrix':  path_matrix }

    def stft_synch(self, choruses, audio_choruses, hop_len=512, win_len=2048,
                   printTimes=False, printError=False, metric='euclidean'):
        chroma_choruses = []
        for chorus in audio_choruses:
            curr_chroma = librosa.feature.chroma_stft(y=chorus, sr=self.sr,
                                                      n_fft=win_len,
                                                      hop_length=hop_len)
            curr_indexes = np.arange(0, curr_chroma.shape[1], dtype=float)
            curr_indexes *= hop_len/self.sr

            chroma_choruses.append({'chroma_values': curr_chroma,
                                    'index_second':  curr_indexes})

        curr_result = self.all_seq_dtw(chroma_choruses, choruses,
                                       printTimes=printTimes, printError=printError,
                                       metric=metric, method='stft', hop_len=str(hop_len))

        return curr_result


    def cqt_synch(self, choruses, audio_choruses, hop_len=512,
                   printTimes=False, printError=False, metric='euclidean'):
        chroma_choruses = []
        for chorus in audio_choruses:
            curr_chroma = librosa.feature.chroma_cqt(y=chorus, sr=self.sr,
                                                     hop_length=hop_len)
            # print('curr_chroma: {}'.format(curr_chroma.shape))
            curr_indexes = np.arange(0, curr_chroma.shape[1], dtype=float)
            curr_indexes *= hop_len/self.sr

            chroma_choruses.append({'chroma_values': curr_chroma,
                                    'index_second':  curr_indexes})

        curr_result = self.all_seq_dtw(chroma_choruses, choruses,
                                       printTimes=printTimes, printError=printError,
                                       metric=metric, method='cqt', hop_len=str(hop_len))

        return curr_result


    def tonnetz_synch(self, choruses, audio_choruses, hop_len=512,
                   printTimes=False, printError=False, metric='euclidean'):
        tonnetz_choruses = []
        for chorus in audio_choruses:
            curr_chroma = librosa.feature.chroma_cqt(y=chorus, sr=self.sr,
                                                     hop_length=hop_len)
            curr_tonnetz = librosa.feature.tonnetz(chroma=curr_chroma,
                                                   sr=self.sr)
            # print('curr_chroma: {}'.format(curr_chroma.shape))
            curr_indexes = np.arange(0, curr_tonnetz.shape[1], dtype=float)
            curr_indexes *= hop_len/self.sr

            tonnetz_choruses.append({'chroma_values': curr_tonnetz,
                                    'index_second':  curr_indexes})

        curr_result = self.all_seq_dtw(tonnetz_choruses, choruses,
                                       printTimes=printTimes, printError=printError,
                                       metric=metric, method='tonnetz', hop_len=str(hop_len))

        return curr_result

    def HFC(self, S):
        x, y = S.shape
        value = np.square(S)
        value = value.T * np.linspace(0, x, x)
        HFC = np.sum(value.T, axis=0)
        return np.abs(np.nan_to_num(HFC))

    def normalize(self, data):
        max_val = np.max(data)
        min_val = np.min(data)
        return (data - min_val)/(max_val - min_val)

    def mixed_features_synch(self, choruses, audio_choruses, hop_len=512,
                             printTimes=False, printError=False, metric='euclidean',
                             feature_list=['cqt'], bass_beat=False):

        bass_range = [32, 261]
        mid_range =  [130, 1046]
        treble_range = [523, 4186]

        chroma_choruses = []
        features_dict = {}

        if 'MFCC' in feature_list:
            features_dict['MFCC'] = []
        if 'cqt' in feature_list:
            features_dict['cqt'] = []
        if 'HFC' in feature_list:
            features_dict['HFC'] = []
        if 'tonnetz' in feature_list:
            features_dict['tonnetz'] = []
        if 'cqt[bass]' in feature_list or 'cqt[ranges]' in feature_list:
            features_dict['cqt[bass]'] = []
        if 'tonnetz[bass]' in feature_list or 'tonnetz[ranges]' in feature_list:
            features_dict['tonnetz[bass]'] = []
        if 'cqt[mid]' in feature_list or 'cqt[ranges]' in feature_list:
            features_dict['cqt[mid]'] = []
        if 'tonnetz[mid]' in feature_list or 'tonnetz[ranges]' in feature_list:
            features_dict['tonnetz[mid]'] = []
        if 'cqt[treble]' in feature_list or 'cqt[ranges]' in feature_list:
            features_dict['cqt[treble]'] = []
        if 'tonnetz[treble]' in feature_list or 'tonnetz[ranges]' in feature_list:
            features_dict['tonnetz[treble]'] = []

        for i, chorus in enumerate(audio_choruses):
            cqt = librosa.core.cqt(chorus, sr=self.sr, hop_length=hop_len)

            if 'MFCC' in feature_list:
                mfcc = librosa.feature.mfcc(sr=self.sr, y=chorus,
                                            hop_length=hop_len, n_mfcc=20)
                features_dict['MFCC'].append(mfcc)
                # print('shape of mfcc is {}'.format(mfcc.shape))

            if 'cqt' in feature_list:
                chroma = np.real(librosa.feature.chroma_cqt(C=cqt, sr=self.sr,
                                                         hop_length=hop_len))
                features_dict['cqt'].append(chroma)
                # print('shape of chroma is {}'.format(chroma.shape))

            if 'HFC' in feature_list:
                HFC = self.HFC(cqt)
                HFC = HFC.reshape((1,len(HFC)))
                features_dict['HFC'].append(HFC)

            if 'tonnetz' in feature_list:
                tonnetz_ = np.real(librosa.feature.chroma_cqt(C=cqt, sr=self.sr,
                                                         hop_length=hop_len))
                tonnetz = librosa.feature.tonnetz(chroma=tonnetz_,
                                                       sr=self.sr)
                features_dict['tonnetz'].append(tonnetz)
                # print('shape of tonnetz is {}'.format(tonnetz.shape))

            if 'cqt[bass]' in feature_list or 'cqt[ranges]' in feature_list:
                chorus_filt = self.filt_range(chorus, bass_range)
                cqt = librosa.core.cqt(chorus_filt, sr=self.sr, hop_length=hop_len)

                # if bass_beat:
                #     beats = [x - choruses[str(i)]['start'] for x in choruses[str(i)]['beats']]
                #     cqt = self.get_bass_aligned_cqt(cqt, beats, hop_len)

                chroma = np.real(librosa.feature.chroma_cqt(C=cqt, sr=self.sr))
                features_dict['cqt[bass]'].append(chroma)

            if 'cqt[mid]' in feature_list or 'cqt[ranges]' in feature_list:
                chorus_filt = self.filt_range(chorus, mid_range)
                cqt = librosa.core.cqt(chorus_filt, sr=self.sr, hop_length=hop_len)

                # if bass_beat:
                #     beats = [x - choruses[str(i)]['start'] for x in choruses[str(i)]['beats']]
                #     cqt = self.get_bass_aligned_cqt(cqt, beats, hop_len)

                chroma = np.real(librosa.feature.chroma_cqt(C=cqt, sr=self.sr))
                features_dict['cqt[mid]'].append(chroma)

            if 'cqt[treble]' in feature_list or 'cqt[ranges]' in feature_list:
                chorus_filt = self.filt_range(chorus, treble_range)
                cqt = librosa.core.cqt(chorus_filt, sr=self.sr, hop_length=hop_len)

                # if bass_beat:
                #     beats = [x - choruses[str(i)]['start'] for x in choruses[str(i)]['beats']]
                #     cqt = self.get_bass_aligned_cqt(cqt, beats, hop_len)

                chroma = np.real(librosa.feature.chroma_cqt(C=cqt, sr=self.sr))
                features_dict['cqt[treble]'].append(chroma)

            if 'tonnetz[bass]' in feature_list or 'tonnetz[ranges]' in feature_list:
                chorus_filt = self.filt_range(chorus, bass_range)
                cqt = librosa.core.cqt(chorus_filt, sr=self.sr, hop_length=hop_len)

                # if bass_beat:
                #     beats = [x - choruses[str(i)]['start'] for x in choruses[str(i)]['beats']]
                #     cqt = self.get_bass_aligned_cqt(cqt, beats, hop_len)

                chroma = np.real(librosa.feature.chroma_cqt(C=cqt, sr=self.sr))
                tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=self.sr)
                features_dict['tonnetz[bass]'].append(tonnetz)

            if 'tonnetz[mid]' in feature_list or 'tonnetz[ranges]' in feature_list:
                chorus_filt = self.filt_range(chorus, mid_range)
                cqt = librosa.core.cqt(chorus_filt, sr=self.sr, hop_length=hop_len)

                # if bass_beat:
                #     beats = [x - choruses[str(i)]['start'] for x in choruses[str(i)]['beats']]
                #     cqt = self.get_bass_aligned_cqt(cqt, beats, hop_len)

                chroma = np.real(librosa.feature.chroma_cqt(C=cqt, sr=self.sr))
                tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=self.sr)
                features_dict['tonnetz[mid]'].append(tonnetz)

            if 'tonnetz[treble]' in feature_list or 'tonnetz[ranges]' in feature_list:
                chorus_filt = self.filt_range(chorus, treble_range)
                cqt = librosa.core.cqt(chorus_filt, sr=self.sr, hop_length=hop_len)

                # if bass_beat:
                #     beats = [x - choruses[str(i)]['start'] for x in choruses[str(i)]['beats']]
                #     cqt = self.get_bass_aligned_cqt(cqt, beats, hop_len)

                chroma = np.real(librosa.feature.chroma_cqt(C=cqt, sr=self.sr))
                tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=self.sr)
                features_dict['tonnetz[treble]'].append(tonnetz)

        if 'cqt[ranges]' in feature_list:
            feature_list.remove('cqt[ranges]')
            feature_list.append('cqt[bass]')
            feature_list.append('cqt[mid]')
            feature_list.append('cqt[treble]')
        if 'tonnetz[ranges]' in feature_list:
            feature_list.remove('tonnetz[ranges]')
            feature_list.append('tonnetz[bass]')
            feature_list.append('tonnetz[mid]')
            feature_list.append('tonnetz[treble]')

        for i in range(len(audio_choruses)):
            feature_set = None
            for curr_feature in feature_list:
                if feature_set is None:
                    feature_set = self.normalize(features_dict[curr_feature][i])
                else:
                    # print('\n{} curr_feature size: {}, all_feature_size: {}\n'.format(curr_feature, features_dict[curr_feature][i].shape, feature_set.shape))
                    feature_set = np.append(feature_set,
                        self.normalize(features_dict[curr_feature][i]), axis=0)

            curr_indexes = np.arange(0, feature_set.shape[1], dtype=float)
            curr_indexes *= hop_len/self.sr

            chroma_choruses.append({'chroma_values': feature_set,
                                    'index_second':  curr_indexes})

        feature_names = '_'.join(sorted(feature_list))
        # print('\nfeature_names: {}\n'.format(feature_names))
        curr_result = self.all_seq_dtw(chroma_choruses, choruses,
                                       printTimes=printTimes, printError=printError,
                                       metric=metric, method=feature_names, hop_len=str(hop_len))
        return curr_result

    def filt_range(self, audio, range):
            b, a = scipy.signal.butter(2, (range[0]/(self.sr/2), range[1]/(self.sr/2)),
                                   btype='bandpass', analog=False,
                                   output='ba')
            audio_filt = scipy.signal.lfilter(b, a, audio)
            return audio_filt

    def get_bass_aligned_cqt(self, cqt, beats, hop_len):
        cqt_times = [x*hop_len/self.sr for x in range(cqt.shape[1])]
        beats_cqt = cqt[:, 0].reshape((cqt.shape[0], 1))

        beats_index = 1
        curr_cqt = cqt[:, 0].reshape((cqt.shape[0], 1))


        for i, cqt_time in enumerate(cqt_times):
            if not i:
                continue
            if beats_index < len(beats)-1:
                curr_beat = beats[beats_index]
            else:
                curr_beat = 1e4
            if cqt_time >= curr_beat:
                curr_cqt = cqt[:, i].reshape((cqt.shape[0], 1))
                beats_index += 1
            beats_cqt = np.concatenate((beats_cqt, curr_cqt), axis=1)

        return beats_cqt

    def custom_synch(self, choruses, audio_choruses, onsets,
                   printTimes=False, printError=False, metric='euclidean',
                   method='stft'):
        chroma_choruses = []
        for chorus, seg_onsets in zip(audio_choruses, onsets):
            curr_chroma = []

            seg_onsets = np.append(seg_onsets, [len(chorus)/self.sr])

            for i in range(len(seg_onsets)-1):
                start = int(seg_onsets[i]*self.sr)
                end = int(seg_onsets[i+1]*self.sr)
                curr_frame = chorus[start:end]
                if 'tonnetz' in method:
                    curr_tonnetz = librosa.feature.chroma_cqt(sr=self.sr, y=curr_frame,
                                                        hop_length=len(curr_frame))
                    chroma = librosa.feature.tonnetz(chroma=curr_tonnetz,
                                                     sr=self.sr)
                elif 'cqt' in method:
                    chroma = librosa.feature.chroma_cqt(sr=self.sr, y=curr_frame,
                                                        hop_length=len(curr_frame))
                else:
                    chroma = librosa.feature.chroma_stft(sr=self.sr, y=curr_frame,
                                                         n_fft=len(curr_frame),
                                                         hop_length=len(curr_frame))
                curr_chroma.append(np.mean(chroma, axis=1))
                # print('method {} y chroma {}'.format(method, curr_chroma[-1].shape), end='\r')


            curr_chroma = np.array(curr_chroma).T

            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 4))
            # librosa.display.specshow(curr_chroma, y_axis='chroma', x_axis='time')
            # plt.colorbar()
            # plt.title('Chromagram')
            # plt.tight_layout()
            # plt.show()

            curr_indexes = seg_onsets
            # print('curr_chroma: {}'.format(curr_chroma.shape))
            chroma_choruses.append({'chroma_values': curr_chroma,
                                    'index_second':  curr_indexes})

        curr_result = self.all_seq_dtw(chroma_choruses, choruses,
                                       printTimes=printTimes, printError=printError,
                                       metric=metric, method='onset'+'_'+method, hop_len=str(hop_len))

        return curr_result


    def save_results(self, db_error, path, dbName):
        with open(path + dbName + '_errors.pkl', 'wb') as f:
            pickle.dump(db_error, f)


def load_results(path, dbName, metric, prefix):
    # path += dbName + '/'
    if len(prefix) and '_' not in prefix[0]:
        prefix = '_' + prefix
    if len(dbName):
        dbName += '_'
    with open(path + dbName + metric+ '_errors'+prefix+'.pkl', 'rb') as f:
        db_error = pickle.load(f)

    return db_error



def get_dataFrame(db_error, dbName,metric):
    total_error = {}
    for config, songs in db_error.items():

        # print('a')
        # print(songs.keys())
        # print('b')
        config_error = 0
        max_error = 0
        min_error = 10
        for song_name, song_error in songs.items():
            curr_error = song_error['mean_error']
            config_error += curr_error
            max_error = max(curr_error, max_error)
            min_error = min(curr_error, min_error)

        total_error[config] = [dbName, metric,
                               min_error, max_error,
                               config_error/len(songs.keys())]

    pd_results = pd.DataFrame(total_error, index=['DB', 'Metric', 'Min',
                                                  'Max', 'Mean']).T

    for index, row in pd_results.iterrows():
        if 'HFC' in str(index):
            pd_results = pd_results.drop(index, axis=0)

    return pd_results



def get_average_results(path, dbName,metric, prefix):
    db_error = load_results(path, dbName, metric, prefix)

    pd_results = get_dataFrame(db_error, dbName,metric)
    pd.set_option('display.float_format', '{:.2e}'.format)

    # for index, row in pd_results.iterrows():
    #     if 'HFC' in str(index) or 'MFCC' in str(index):
    #         # print(index)
    #         pd_results = pd_results.drop(index, axis=0)


    print(pd_results.sort_values('Mean'))
    # sorted_values = sorted(total_error.items(), key=lambda kv: kv[1])
    # for key, value in sorted_values:
    #     print('For DataBase {} error of {} with metric {} has min: {:.2e}, max: {:.2e} and meand: {:.2e}'.format(dbName,
    #                                         key, metric,max_min_error[key+'_min'],max_min_error[key+'_max'],value), end='\n')

    # print(total_error)


def get_comparations(path, dbName,metric, prefix,
                     feature_combinations):

    db_error = load_results(path, dbName, metric, prefix)

    pd_results = get_dataFrame(db_error, dbName,metric)
    pd.set_option('display.float_format', '{:.2e}'.format)

    hop_2048 = []
    hop_4096 = []
    hop_8192 = []

    pure_cqt = []
    only_bass_cqt = []
    only_mid_cqt = []
    bass_mid_cqt = []
    ranges_cqt = []

    pure_tonnetz = []
    only_bass_tonnetz = []
    only_mid_tonnetz = []
    bass_mid_tonnetz = []
    ranges_tonnetz = []

    with_MFCC = []
    without_MFCC = []

    cqt_no_tonnetz = []
    tonnetz_no_cqt = []

    all_features = []

    for index, row in pd_results.iterrows():
        for feature_set in feature_combinations:
            all_features.append(row.loc['Mean'])

            if feature_set + '_hop2048'  == str(index):
                hop_2048.append(row.loc['Mean'])
                continue
            elif feature_set + '_hop4096'  == str(index):
                hop_4096.append(row.loc['Mean'])
                continue
            elif feature_set + '_hop8192'  == str(index):
                hop_8192.append(row.loc['Mean'])
                continue

            if 'MFCC' in str(index):
                with_MFCC.append(row.loc['Mean'])
            else:
                without_MFCC.append(row.loc['Mean'])

            if 'cqt_' in str(index):
                pure_cqt.append(row.loc['Mean'])
            if 'cqt[bass]' in str(index) and 'cqt[mid]' not in str(index):
                only_bass_cqt.append(row.loc['Mean'])
            if 'cqt[bass]' not in str(index) and 'cqt[mid]' in str(index):
                only_mid_cqt.append(row.loc['Mean'])
            if 'cqt[bass]' in str(index) and 'cqt[mid]' in str(index):
                bass_mid_cqt.append(row.loc['Mean'])
            if 'cqt[ranges]' in str(index):
                ranges_cqt.append(row.loc['Mean'])

            if 'tonnetz_' in str(index):
                pure_tonnetz.append(row.loc['Mean'])
            if 'tonnetz[bass]' in str(index) and 'tonnetz[mid]' not in str(index):
                only_bass_tonnetz.append(row.loc['Mean'])
            if 'tonnetz[bass]' not in str(index) and 'tonnetz[mid]' in str(index):
                only_mid_tonnetz.append(row.loc['Mean'])
            if 'tonnetz[bass]' in str(index) and 'tonnetz[mid]' in str(index):
                bass_mid_tonnetz.append(row.loc['Mean'])
            if 'tonnetz[ranges]' in str(index):
                ranges_tonnetz.append(row.loc['Mean'])

            if 'cqt' in str(index) and 'tonnetz' not in str(index):
                cqt_no_tonnetz.append(row.loc['Mean'])
            elif 'tonnetz' in str(index) and 'cqt' not in str(index):
                tonnetz_no_cqt.append(row.loc['Mean'])


    all_features = sum(all_features) / len(all_features)

    hop_2048 = sum(hop_2048) / len(hop_2048)
    hop_4096 = sum(hop_4096) / len(hop_4096)
    hop_8192 = sum(hop_8192) / len(hop_8192)

    pure_cqt = sum(pure_cqt) / len(pure_cqt)
    only_bass_cqt = sum(only_bass_cqt) / len(only_bass_cqt)
    only_mid_cqt = sum(only_mid_cqt) / len(only_mid_cqt)
    bass_mid_cqt = sum(bass_mid_cqt) / len(bass_mid_cqt)
    ranges_cqt = sum(ranges_cqt) / len(ranges_cqt)

    pure_tonnetz = sum(pure_tonnetz) / len(pure_tonnetz)
    only_bass_tonnetz = sum(only_bass_tonnetz) / len(only_bass_tonnetz)
    only_mid_tonnetz = sum(only_mid_tonnetz) / len(only_mid_tonnetz)
    bass_mid_tonnetz = sum(bass_mid_tonnetz) / len(bass_mid_tonnetz)
    ranges_tonnetz = sum(ranges_tonnetz) / len(ranges_tonnetz)

    with_MFCC = sum(with_MFCC) / len(with_MFCC)
    without_MFCC = sum(without_MFCC) / len(without_MFCC)

    cqt_no_tonnetz = sum(cqt_no_tonnetz) / len(cqt_no_tonnetz)
    tonnetz_no_cqt = sum(tonnetz_no_cqt) / len(tonnetz_no_cqt)

    print('    Overall Mean: {}'.format(all_features))
    print('\n    mean 2048: {0:.4f}, mean 4096: {1:.4f}, mean 8192: {2:.4f}'.format(hop_2048, hop_4096, hop_8192))
    print('        2048/4096: {0:.3f}, difference of {1:.1f}%'.format(hop_2048/hop_4096, (1-hop_2048/hop_4096)*100))
    print('        2048/8192: {0:.3f}, difference of {1:.1f}%'.format(hop_2048/hop_8192, (1-hop_2048/hop_8192)*100))
    print('        4096/8192: {0:.3f}, difference of {1:.1f}%'.format(hop_4096/hop_8192, (1-hop_4096/hop_8192)*100))

    print('\n    mean pure_cqt: {0:.4f}, mean cqt[bass]: {1:.4f}, mean cqt[mid]: {2:.4f} mean cqt[bass+mid]: {3:.4f}, mean cqt[ranges]: {4:.4f}'.format(pure_cqt, only_bass_cqt, only_mid_cqt, bass_mid_cqt, ranges_cqt))
    print('        ranges/pure cqt: {0:.3f}, difference of {1:.1f}%'.format(ranges_cqt/pure_cqt, (1-ranges_cqt/pure_cqt)*100))
    print('        ranges/bass cqt: {0:.3f}, difference of {1:.1f}%'.format(ranges_cqt/only_bass_cqt, (1-ranges_cqt/only_bass_cqt)*100))
    print('        ranges/mid cqt: {0:.3f}, difference of {1:.1f}%'.format(ranges_cqt/only_mid_cqt, (1-ranges_cqt/only_mid_cqt)*100))
    print('        ranges/bass_mid cqt: {0:.3f}, difference of {1:.1f}%'.format(ranges_cqt/bass_mid_cqt, (1-ranges_cqt/bass_mid_cqt)*100))

    print('\n    mean pure_tonnetz: {0:.4f}, mean tonnetz[bass]: {1:.4f}, mean tonnetz[mid]: {2:.4f} mean tonnetz[bass+mid]: {3:.4f}, mean tonnetz[ranges]: {4:.4f}'.format(pure_tonnetz, only_bass_tonnetz, only_mid_tonnetz, bass_mid_tonnetz, ranges_tonnetz))
    print('        ranges/pure tonnetz: {0:.3f}, difference of {1:.1f}%'.format(ranges_tonnetz/pure_tonnetz, (1-ranges_tonnetz/pure_tonnetz)*100))
    print('        ranges/bass tonnetz: {0:.3f}, difference of {1:.1f}%'.format(ranges_tonnetz/only_bass_tonnetz, (1-ranges_tonnetz/only_bass_tonnetz)*100))
    print('        ranges/mid tonnetz: {0:.3f}, difference of {1:.1f}%'.format(ranges_tonnetz/only_mid_tonnetz, (1-ranges_tonnetz/only_mid_tonnetz)*100))
    print('        ranges/bass_mid tonnetz: {0:.3f}, difference of {1:.1f}%'.format(ranges_tonnetz/bass_mid_tonnetz, (1-ranges_tonnetz/bass_mid_tonnetz)*100))

    print('\n    mean with MFCC {0:.4f}, mean without MFCC: {1:.4f}'.format(with_MFCC, without_MFCC))
    print('        with/without MFCC: {0:.3f}, difference of {1:.1f}%'.format(with_MFCC/without_MFCC, (1-with_MFCC/without_MFCC)*100))

    print('\n    mean only with cqt {0:.4f}, mean only with tonnetz: {1:.4f}'.format(cqt_no_tonnetz, tonnetz_no_cqt))
    print('        cqt/tonnetz : {0:.3f}, difference of {1:.1f}%'.format(cqt_no_tonnetz/tonnetz_no_cqt, (1-cqt_no_tonnetz/tonnetz_no_cqt)*100))






#
