# MIR imports
import librosa
import madmom
from madmom.audio.filters import MelFilterbank

# self imports
from   dlUtils           import functional_both, only_decoder
from   dlUtils           import  chordTesting, chordEval
from   keras             import backend as K
from   keras.backend     import tensorflow_backend
import keras
from   keras.layers      import Input
import tensorflow        as     tf

# Pyhton imports
import random
import os
import pickle
import glob
import json
import time
import sys
import scipy.io.wavfile
import numpy     as np
import pandas    as pd
from collections import deque
import matplotlib.pyplot             as plt
from matplotlib.backends.backend_pdf import PdfPages


# Own Code Imports
sys.path.insert(0, '../')
sys.path.insert(0, '../segmentSynch/')
from audioData import AudioData
from synchronizationManager import SynchronizationData


class transcriptionData(SynchronizationData):
    """
        Inherits methods read_json(path), _get_JAAH_choruses(json_data) and
        get_audio_JAAH_choruses(audio, json_path, song_name)
    """

    def __init__(self, sr=44100, workin_dir='.', hop_len=512,
                 dtw_results_path=''):

        super().__init__(sr=sr, workin_dir=workin_dir)

        self.dtw_results_path = dtw_results_path

        self.root_coding =  deque([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Relative to C
        self.type_template = {
                                'maj' : deque([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
                                '7'   : deque([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
                                'min' : deque([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
                                'hdim': deque([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]),
                                'dim' : deque([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
                            }

        self.rotation_index = {
                                'C':  0,
                                'C#': 1,
                                'Db': 1,
                                'D':  2,
                                'D#': 3,
                                'Eb': 3,
                                'E':  4,
                                'F':  5,
                                'F#': 6,
                                'Gb': 6,
                                'G':  7,
                                'G#': 8,
                                'Ab': 8,
                                'A':  9,
                                'A#': 10,
                                'Bb': 10,
                                'B':  11,
                                'Cb': 11
                            }

    def getCqtChoruses(self,audio_choruses, info={}, hop_len=1024, method='cqt'):
        """
            Gets the CQT of each chorus in audio_choruses. Returns A list in
            the same order that the input audio_choruses.

        """
        if 'tuning' in info.keys():
            tuning = info['tuning']
            if not isinstance(tuning, (int, float, complex)):
                tuning = None
            else:
                # print('tuning: {}  -> {}'.format(tuning, 1200 * np.log2(tuning/440) / 100))
                tuning = 1200 * np.log2(tuning/440) / 100
        else:
            tuning = None

        cqt_choruses = []

        for curr_audio in audio_choruses:
            if 'cqt' in method:
                curr_cqt = AudioData.getCQT(curr_audio, hop_len=hop_len, tuning=tuning)
            elif 'mel' in method:
                newCqt1 = AudioData.getMelSpectrogram(curr_audio, hop_len=hop_len, win_len=2048, tuning_freq=tuning, n_bands=85)
                newCqt2 = AudioData.getMelSpectrogram(curr_audio, hop_len=hop_len, win_len=4096, tuning_freq=tuning, n_bands=85)
                newCqt3 = AudioData.getMelSpectrogram(curr_audio, hop_len=hop_len, win_len=8192, tuning_freq=tuning, n_bands=85)
                curr_cqt = np.empty((newCqt1.shape[1], newCqt1.shape[0], 3), dtype=float)
                curr_cqt[:,:,0] = newCqt1.T
                curr_cqt[:,:,1] = newCqt2.T
                curr_cqt[:,:,2] = newCqt3.T
            # print('Shape del cqt: {}'.format(curr_cqt.shape))
            cqt_choruses.append(curr_cqt)

        return cqt_choruses

    # def pitch_shift(self, path, i):
    #     i = str(i)
    #     command = 'rubberband -p '+i+' --pitch-hq '+path+' '+'temp.wav'
    #     os.system(command)

    def getDtwPaths(self, song_name):
        print('song_name ' +  song_name)
        if len(self.dtw_results_path) < 2:
            base_path = '../../results/segmentSynch/JAAH_results/paths_beats/'
        else:
            base_path = self.dtw_results_path
        dtw_paths = []
        for i in range(1, 1000):
            curr_path = song_name+'_0_'+str(i)+'_MFCC_cqt[bass]_cqt[mid]_cqt[treble]_tonnetz_2048_mahalanobis.pkl'
            print(base_path + curr_path)
            if not os.path.isfile(base_path + curr_path):
                break

            file = open(base_path + curr_path, 'rb')
            curr_dtw = pickle.load(file)
            file.close()

            dtw_paths.append(curr_dtw['dtw_path'])

        return dtw_paths

    def getChords(self, choruses):
        measure_chords = []
        final_chords = {}
        for _, value in choruses.items():
            if not isinstance(value, dict):
                continue
            curr_chords = self._parseChords(value['chords'])
            measure_chords.append(curr_chords)

        average_chords = {}
        for i in range(len(curr_chords)):
            average_chords[i] = []
            for curr_chord in measure_chords:
                average_chords[i].append(curr_chord[i])

        for key, measure_chord_list in average_chords.items():
            final_chords[key] = self._getMostProbableChord(measure_chord_list)

        return final_chords

    def alignChordsBeats(self, chords, chorus, song_info,
                         sr=44100, hop_len=2048):

        beats = list(np.array(chorus['beats']) - chorus['start'])
        metric = song_info['metric']
        num = int(metric.split('/')[0])
        beat_labels = np.zeros((len(np.arange(0, 2*beats[-1] - beats[-2], hop_len/sr))), dtype=int)

        assert len(chords.keys()) == len(beats)/num
        # print('    arange beats len:  {}'.format(len(np.arange(0, 2*beats[-1] - beats[-2], hop_len/sr))))


        # Dict of beat index and the corresponding chords
        beat_chords = {}
        for i in range(len(beats)):
            beat_chords[i] = None

        # assign chord to beat
        beat_ind = 0
        for _, curr_chords in chords.items():
            n_chords = len(curr_chords)
            if n_chords == 1:
                beat_chords[beat_ind] = curr_chords[0]
                beat_ind += num
            elif n_chords == 2:
                for i in [0, 1]:
                    beat_chords[beat_ind] = curr_chords[i]
                    beat_ind += int(num/2)
            else:
                for i in [0, 1, 2, 3]:
                    beat_chords[beat_ind] = curr_chords[i]
                    beat_ind += int(num/4)

        chord_labels = []
        beat_ind = 0
        curr_chord = beat_chords[0]
        beats.append(2*beats[-1] - beats[-2])
        beat_labels[0] = 1
        # print('beats[-4:] {}'.format(beats[-4:]))
        for i, curr_time in enumerate(np.arange(0, beats[-1], hop_len/sr)):
            # print('curr_time {}, curr_beat: {}, beat_ind {}, curr_chord: {}'.format(curr_time.round(2), beats[beat_ind].round(2), beat_ind, curr_chord))
            chord_labels.append(curr_chord)


            if curr_time >= beats[beat_ind]:
                if i > 1:
                    beat_labels[i+1] = 1
                beat_ind += 1
                if beat_chords[beat_ind-1] is  not None:
                    curr_chord = beat_chords[beat_ind-1]

        # for c_chord, c_beat in zip(chord_labels, beat_labels):
        #     print('{} - {}'.format(c_chord, c_beat))

        return chord_labels, beat_labels


    def codeChordLabels(self, aligned_chords, transpose=0):
        """
            Creates one-hot labels for roots and notes in a chord. Can transpose
            'i' semitones
        """

        coded_root = np.zeros((13, len(aligned_chords)), dtype=int)
        coded_notes = np.zeros((12, len(aligned_chords)), dtype=int)

        for i, curr_chord in enumerate(aligned_chords):
            root, type = curr_chord.split(':')

            if 'N' != root:
                rot_ind = self.rotation_index[root] + transpose
                curr_root = self.root_coding.copy()
                curr_root.rotate(rot_ind)
                curr_notes = self.type_template[type].copy()
                curr_notes.rotate(rot_ind)
                curr_root.append(0)
                # curr_notes.append(0)
            else:
                curr_root = np.zeros((13))
                curr_root[-1] = 1
                curr_notes = np.zeros((12))
                # curr_notes[-1] = 1


            root_list = list(curr_root.copy())
            note_list = list(curr_notes.copy())

            coded_root[:,i] = root_list
            coded_notes[:,i] = note_list


        # for i, curr_chord in enumerate(aligned_chords):
        #     print('    Chord {}'.format(curr_chord))
        #     print('        coded_root: {}'.format(coded_root[:,i]))
        #     print('        coded_notes: {}'.format(coded_notes[:,i]))

        return coded_root, coded_notes



    def averageCqts(self, warped_cqts):
        # print('input {} sequences of shape {}'.format(len(warped_cqts), warped_cqts[0].shape))
        if len(warped_cqts[0].shape) < 3:
            np_cqts = np.zeros((*warped_cqts[0].shape, len(warped_cqts)))
            for i, curr_cqt in enumerate(warped_cqts):
                np_cqts[:,:,i] = curr_cqt
            averaged_cqts = np_cqts.mean(axis=2)
        else:
            averaged_cqts = np.zeros((*warped_cqts[0].shape[:2], 3))
            for i in range(3):
                np_cqts = np.zeros((*warped_cqts[0].shape[:2], len(warped_cqts)))
                for j, curr_cqt in enumerate(warped_cqts):
                    np_cqts[:,:,j] = curr_cqt[:,:,i]
                averaged_cqts[:,:,i] = np_cqts.mean(axis=2)

        # print('output has shape: {}'.format(averaged_cqts.shape))
        return averaged_cqts

    def _simplifyChord(self, chord):
        """
            Only a dict of maj, 7, min, hdim and dim chord classes is used.
        """

        splitted_chords = chord.rstrip().lstrip().split(' ')
        corrected_chord = []

        for curr_chord in splitted_chords:
            if ':' in curr_chord:
                root = curr_chord.split(':')[0].split('/')[0]
                if 'hdim' in curr_chord:
                    corrected_chord.append(root + ':hdim')
                elif 'dim' in curr_chord:
                    corrected_chord.append(root + ':dim')
                elif 'min' in curr_chord:
                    corrected_chord.append(root + ':min')
                elif '7' in curr_chord and 'maj' not in curr_chord:
                    corrected_chord.append(root + ':7')
                else:
                    corrected_chord.append(root + ':maj')
            else:
                # print('  this has no descprition...')
                root = curr_chord.split('/')[0].replace(' ', '')
                if len(root):
                    corrected_chord.append(root + ':maj')

        # print('    Chord {} ---> {}'.format(chord, corrected_chord))
        return corrected_chord


    def _getMostProbableChord(self, chord_list):
        """
            If a chord is rendered differently within choruses, the mode is taken.
        """

        chord_dict = {i:chord_list.count(i) for i in chord_list}
        max_chord = max(chord_dict, key=chord_dict.get)
        return self._simplifyChord(max_chord)

    def _parseChords(self, chords):
        all_chords = ''
        for curr_chord in chords:
            all_chords = all_chords + curr_chord[1:]
        chord_list = all_chords.split('|')[:-1]

        return chord_list

    def warpCqts(self, cqts, paths):

        assert len(cqts) == len(paths) + 1
        warped_cqts = [cqts[0]]

        for curr_cqt, curr_path in zip(cqts[1:], paths):
            warped_cqts.append(self._warpCurrentCqt(curr_cqt, curr_path))

        for i, curr_cqt in enumerate(warped_cqts):
            #assert np.array_equal(cqts[0].shape, curr_cqt.shape), '    cqt at ' + str(i) + ' should have ' + str(cqts[0].shape) + ' but has ' + str(curr_cqt.shape)
            if not np.array_equal(cqts[0].shape, curr_cqt.shape):
                 print('    cqt at ' + str(i) + ' should have ' +
                        str(cqts[0].shape) + ' but has ' + str(curr_cqt.shape))
        return warped_cqts


    def _warpCurrentCqt(self, cqt, path):
        """
            Warps the input cqt expanding and compressing according to the
            warping path. The first column of the path is the reference and the
            second corresponds to the input cqt, so the second column is warped
            to match the first.

            :param cqt: input cqt to be warped. Shape must be frequencyBins x Frames
            :type cqt: np.array
            :param path: input warping path
            :path type: np.array
            :return: np.array, warped cqt
        """

        expand, compress = self._getExpandCompress(path)
        #newCqt = cqt.copy()
        if len(cqt.shape) < 3:
            newCqt = self._expandCompressCqt(cqt, expand, compress)
        else:
            currCqt1 = self._expandCompressCqt(cqt[:,:,0], expand, compress)
            currCqt2 = self._expandCompressCqt(cqt[:,:,1], expand, compress)
            currCqt3 = self._expandCompressCqt(cqt[:,:,2], expand, compress)
            newCqt = np.empty((*currCqt1.shape, 3), dtype=float)
            newCqt[:,:,0] = currCqt1
            newCqt[:,:,1] = currCqt2
            newCqt[:,:,2] = currCqt3

        # print('Shape del cqt warped: {}'.format(newCqt.shape))

        return newCqt

    def _expandCompressCqt(self, cqt, expand, compress):
        for index, times in reversed(sorted(compress.items())):
            cols = cqt[:, index:index+times+1]
            cqt = np.delete(cqt, range(index, index+times+1), 1)
            meanCols = np.mean(cols, axis=1)
            cqt = np.insert(cqt, index, meanCols, axis=1)

        expand = self._updateExpandIndexes(expand, compress)

        for index, times in reversed(sorted(expand.items())):

            if index < cqt.shape[1] - 2:
                nc = cqt[:, index:index+2].T
                repCol = scipy.interpolate.pchip_interpolate(np.linspace(0,
                                                                   nc.shape[0],
                                                                   nc.shape[0]),
                                                             nc,
                                                             np.linspace(0,
                                                                   nc.shape[0],
                                                                   nc.shape[0] +
                                                                   times))
                cqt = np.insert(cqt, index, repCol[1:-1, :], axis=1)
            else:
                for _ in range(times):
                    repCol = cqt[:,index]
                    cqt = np.insert(cqt, index, repCol, axis=1)

        return cqt

    def _getExpandCompress(self, path):
        """
            Get the indexes where de sequence has to be expanded and compressed.
            The return values are contained in a Dict, where the key is the
            index and the value the number of frames that have to be expanded
            or compressed.

            :param path: array with the warping path. Elements must be in seconds.
            :type path: np.array
            :return: expand and compress dicts
        """

        path = np.multiply(path, 44100/2048).round(0).astype(int)
        expand = {}
        compress = {}

        prev_i = path[0, 0]
        prev_j = path[0, 1]

        cumulatedComp = 0
        cumulatedExp = 0

        for i, j in path[1:, :]:

            if  prev_i == i and prev_j == j:
                continue

            if prev_i == i:
                cumulatedComp +=1
                if j-cumulatedComp not in compress.keys():
                    compress[j-cumulatedComp] = 1
                else:
                    compress[j-cumulatedComp] += 1
            else:
                cumulatedComp = 0

            if prev_j == j:
                if j-cumulatedExp not in expand.keys():
                    expand[j-cumulatedExp] = 1
                else:
                    expand[j-cumulatedExp] += 1
                cumulatedExp = 1
            else:
                cumulatedExp = 0
            prev_i = i
            prev_j = j
        return expand, compress





    def _updateExpandIndexes(self, expand, compress):
        """
            Method that updates the expand indexes, because they where modified
            after compression.

            :param : Expand dict
            :type : dict
            :param : Compress dict
            :type : dict
            :return: dict. Updated indexes of expand.
        """

        indexes = sorted(expand.keys())
        outIndexes = indexes
        for compIndex, compTimes in sorted(compress.items()):
            for i, currIndex in enumerate(indexes):
                if compIndex < currIndex:
                    outIndexes[i:] = [x-compTimes for x in outIndexes[i:]]
                    break
        outExpand = {}
        for i, itms in enumerate(sorted(expand.items())):
            _, value = itms
            outExpand[outIndexes[i]] = value
        return outExpand



    def predict_chords(self, cqts):
        """
            Perform prediction on previously synchronized audios, that should
            come as time-frequency representations already. The input is a
            list of time-frequency samples. The encoded latent features are averaged
            and final prediction obtained.
        """
        # tensorflow configuration
        config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        session = tf.Session(config=config)
        tensorflow_backend.set_session(session)


        # Hard coded params. Best models obtained work with those...
        both_shape  = (None, 84, 1)
        both_path = '../../models/chordTranscription/both_second/both_second_best.h5'
        both = functional_both(both_shape)
        both.load_weights(both_path)


        encoder = keras.Model(inputs=both.input, outputs=
                                    [both.get_layer('sq_conv4').output,
                                     both.get_layer('root').output,
                                     both.get_layer('notes').output,
                                     both.get_layer('beats').output])

        dec_shape = (None, 109)
        decoder = only_decoder(dec_shape)
        decoder.load_weights(both_path, by_name=True)

        testPred = chordTesting('average_predictions')
        testLatent = chordTesting('average_latent_features')

        CE = chordEval()

        for curr_chorus in cqts:

            test_data = np.zeros((1, *curr_chorus.T.shape, 1))
            test_data[0,:,:,0] = curr_chorus.T

            r_pred, n_pred, b_pred, c_pred = both.predict(test_data)

            l_i_pred, r_i_pred, n_i_pred, b_i_pred = encoder.predict(test_data)


            all_predictions = chordTesting.toHRlabels(r_pred, n_pred, b_pred,
                                                      c_pred, CE)
            pred_roots, pred_notes, pred_beats, pred_chords = all_predictions

            l_interm, r_interm, n_interm, b_interm = encoder.predict(test_data)
            testLatent.stackLatentValues(l_interm, r_pred, n_pred, b_pred)

            # Stack predictions of each chorus
            testPred.stackPredValues(r_pred, n_pred, b_pred, c_pred)


        testLatent.averageLatentStack()
        testPred.averagePredStack()
        # Obtain chords prediction for averaged latent features
        c_pred_decoder = decoder.predict(testLatent.concatenatedLatent)

        r_average = testPred.stack_root
        n_average = testPred.stack_notes
        b_average = testPred.stack_beats
        c_average = testPred.stack_chords

        # Human readable labels for averaged predictions
        average_pred = chordTesting.toHRlabels(r_average, n_average,
                                                b_average, c_average,CE)
        average_latent = chordTesting.toHRlabels(r_average, n_average,
                                               b_average, c_pred_decoder,CE)

        avPred_chords = average_pred[3]
        avLat_chords = average_latent[3]

        return avPred_chords, avLat_chords



    def alignSequences(self, seq1, seq2, path):
        """
            [Not used!]
            Aligns two audio sequences according to the given path. If the given
            path are the beat sequences, the result is the ground truth alignment.
        """

        path = self._getAnchorPoints(path)
        audio_seg1, audio_seg2 = self._getAudioAnchorSegments(seq1, seq2, path)

        aligned_seq1 = []
        aligned_seq2 = []

        for curr_seq1, curr_seq2 in zip(audio_seg1, audio_seg2):
            if len(curr_seq1) == len(curr_seq2):
                new_seq1 = curr_seq1
                new_seq2 = curr_seq2
            else:
                new_seq1 = curr_seq1
                new_seq2 = pyrb.time_stretch(curr_seq2, self.sr,
                                             len(curr_seq2)/len(curr_seq1))

            aligned_seq1.extend(new_seq1)
            aligned_seq2.extend(new_seq2)


    def _getAudioAnchorSegments(self, audio1, audio2, path):
        """
            [Not used!]
            get audio sub sequences matcing anchor points.
        """
        audioSeg1 = []
        audioSeg2 = []

        currSample = 0
        for currPoint in path[1:, 0]:
            if currSample == int(currPoint*self.sr):
                continue
            audioSeg1.append(seq1[currSample:int(currPoint*self.sr)])
            currSample = int(currPoint*self.sr)

        currSample = 0
        for currPoint in path[1:, 1]:
            if currSample == int(currPoint*self.sr):
                continue
            audioSeg2.append(seq2[currSample:int(currPoint*self.sr)])
            currSample = int(currPoint*self.sr)

        return audioSeg1, audioSeg2

    def _getAnchorPoints(self, path):
        """
           [Not used!]
            Get anchor points to perform time stretching and synch the audio
            sequences.
        """

        anchor_points = []
        prev_i = path[0, 0]
        prev_j = path[0, 1]
        for i, j in path[1:, :]:
            if prev_i == i or prev_j == j:
                prev_i = i
                prev_j = j
                continue
            else:
                anchor_points.append([i,j])
                prev_i = i
                prev_j = j
        return np.array(anchor_points)



class leadSheetClass:

    def __init__(self, num_measures, metre):
        self.num_measures = num_measures
        self.metre = metre
        self.num, self.den = self.metre.split('/')
        self.num = int(self.num)
        self.den = int(self.den)
        self.slots = self._initSlots()
        self.types = set()

    def print(self, path):
        # for key, value in self.slots.items():
        #     print('{} -> {}'.format(key,value))

        lines = int(self.num_measures/self.num)+(self.num_measures%self.num >0)
        grid = np.zeros((5, lines))

        h_space = 0.25
        v_space = 0.1

        beat_space = h_space/(self.num + 1)

        song_name = path.split('/')[-1].replace('_', ' ')
        with PdfPages(path + '.pdf') as pdf:

            fig = plt.figure(figsize=(4, 6))
            ax = fig.add_subplot(111)
            txt = ax.text(0.5, 1.05, song_name,
                          fontSize=13, fontweight='bold',
                          bbox={'facecolor':'white','alpha':1,
                                'edgecolor':'none','pad':1},
                          ha='center', va='center')

            def plot_bar(h,v):
                txt = ax.text(h, v, '|', fontSize=15, fontweight='bold')
                return txt

            for j in range(5):
                for i in range(lines):
                    if not i and not j:
                        txt = plot_bar(j*h_space-0.01, 0.9-i*v_space)
                    txt = plot_bar(j*h_space, 0.9-i*v_space)
            txt = plot_bar(j*h_space+0.01, 0.9-i*v_space)


            num_types = len(self.types)
            self.types = sorted(list(self.types))
            if num_types == 1:
                v_offset = [0]
                colors = ['black']
            elif num_types == 2:
                v_offset = [0.01, -0.01]
                colors = ['black', 'blue']
            elif num_types == 3:
                v_offset = [0.02, 0, -0.02]
                colors = ['black', 'blue', 'red']
            else:
                v_offset = [0.03, 0.01, -0.01, -0.03]
                colors = ['black', 'blue', 'red', 'green']


            for i, type in enumerate(self.types):
                for measure_idx, measure_dict in self.slots.items():
                    v = 0.9 - (measure_idx // self.num) * v_space + v_offset[i]
                    measure_offset = (measure_idx % self.num) * h_space

                    for beat in range(self.num):
                        beat_pos = h_space / 4
                        h = beat * beat_pos +  measure_offset + h_space/8

                        if type in measure_dict[beat].keys():
                            chord = measure_dict[beat][type]
                        elif not beat:
                            pass
                        else:
                            continue
                        root, chord_type = self._simplifyAnnotation(chord)
                        txt = ax.text(h, v, r'$'+root+'_{'+chord_type+'}$',
                                      fontSize=5.5, fontweight='bold',
                                      color=colors[i])


            for i, type in enumerate(self.types):
                 txt = ax.text(0, v_offset[i]*1.6, type,
                               fontSize=10, fontweight='bold',
                               color=colors[i])

            txt.set_clip_on(False)
            plt.axis('off')
            pdf.savefig()

    def _simplifyAnnotation(self, chord):

        chord = chord.replace('B/Cb','B').replace('C#/Db', 'Db')
        chord = chord.replace('D#/Eb','Eb').replace('F#/Gb', 'Gb')
        chord = chord.replace('G#/Ab','Ab').replace('A#/Bb', 'Bb')
        chord = chord.replace('min', '-')#.replace('hdim', r'$\oslash$')
        chord = chord.replace('maj', ' ')#.replace('dim', r'$\bigcirc$')
        chord = chord.replace('C#', 'Db').replace('D#', 'Eb')
        chord = chord.replace('F#', 'Gb').replace('G#', 'Ab').replace('A#','Bb')

        return chord.split(':')


    def _initSlots(self):
        slots = {}
        for measure in range(self.num_measures):
            curr_dict = {}
            for i in range(self.num):
                curr_dict[i] = {}
            slots[measure] = curr_dict

        return slots

    def populateChords(self, aligned_chords, type):

        self.types.add(type)
        if isinstance(aligned_chords, list):
            N = len(aligned_chords)
            beat_times = np.linspace(0, N-1, num=self.num*self.num_measures)

            chord_change = []
            prev_chord = aligned_chords[0]
            chord_list = []
            for i, curr_chord in enumerate(aligned_chords):
                if curr_chord != prev_chord:
                    approx_beat = self._approxBeat(i, beat_times).item()
                    chord_change.append(approx_beat)
                    prev_chord = curr_chord
                    chord_list.append(curr_chord)

            assert len(chord_change) == len(chord_list)


            self.slots[0][0][type] = aligned_chords[0]

            for chord, ind in zip(chord_list, chord_change):
                measure = ind // self.num
                beat = ind % self.num
                self.slots[measure][beat][type] =  chord

        elif isinstance(aligned_chords, dict):
            for measure, chord_list in aligned_chords.items():
                # chord_list = list(dict.fromkeys(chord_list))
                if len(chord_list) == 1:
                    self.slots[measure][0][type] = chord_list[0]
                elif len(chord_list) == 2:
                    self.slots[measure][0][type] = chord_list[0]
                    self.slots[measure][2][type] = chord_list[1]
                else:
                    self.slots[measure][0][type] = chord_list[0]
                    self.slots[measure][1][type] = chord_list[1]
                    self.slots[measure][2][type] = chord_list[2]
                    self.slots[measure][3][type] = chord_list[3]
        else:
            raise TypeError('aligned chords not in proper format')



    def _approxBeat(self, i, beats):
        ind = np.argwhere(beats>=i)[0]

        if not ind:
            return beat[ind]

        # print('beat-1 {}, beat {}, beat+1 {}'.format(beat_times[approx_beat-1], beat_times[approx_beat], beat_times[approx_beat+1]))

        dist1 = abs(beats[ind] - i)
        dist2 = abs(beats[ind-1] - i)
        if ind % 2:
            dist1 += 2
        else:
            dist2 += 2

        if dist1 > dist2:
            return ind-1
        else:
            return ind












        #
