# MIR imports
import librosa
import madmom
from madmom.audio.filters import MelFilterbank

# Pyhton imports
import random
import os
import pickle
import glob
import json
import time
import sys
import scipy.io.wavfile
import numpy    as np
import pandas   as pd

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



    def getCqtChoruses(self,audio_choruses, hop_len=1024):
        """
            Gets the CQT of each chorus in audio_choruses. Returns A list in
            the same order that the input audio_choruses.

        """

        cqt_choruses = []

        for curr_audio in audio_choruses:

            curr_cqt = AudioData.getCQT(curr_audio, hop_len=hop_len)

            cqt_choruses.append(curr_cqt)

        return cqt_choruses

    def getDtwPaths(self, song_name):
        dtw_paths = []
        for i in range(1, 1000):
            curr_path = song_name+'_0_'+str(i)+'_MFCC_cqt[bass]_cqt[mid]_cqt[treble]_tonnetz_2048_mahalanobis.pkl'
            if not os.path.isfile(self.dtw_results_path + curr_path):
                break

            file = open(self.dtw_results_path + curr_path, 'rb')
            curr_dtw = pickle.load(file)
            file.close()

            dtw_paths.append(curr_dtw['dtw_path'])

        return dtw_paths

    def getChords(self, choruses):
        essence_chords = []
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

    def _simplifyChord(self, chord):
        splitted_chords = chord.rstrip().lstrip().split(' ')
        corrected_chord = []

        for curr_chord in splitted_chords:
            if ':' in curr_chord:
                root = curr_chord.split(':')[0]
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
                print('  this has no descprition...')
                corrected_chord.append(curr_chord + ':maj')

        print('    Chord {} ---> {}'.format(chord, corrected_chord))
        return corrected_chord


    def _getMostProbableChord(self, chord_list):

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
        newCqt = cqt.copy()

        for index, times in reversed(sorted(compress.items())):
            cols = newCqt[:, index:index+times+1]
            newCqt = np.delete(newCqt, range(index, index+times+1), 1)
            meanCols = np.mean(cols, axis=1)
            newCqt = np.insert(newCqt, index, meanCols, axis=1)

        expand = self._updateExpandIndexes(expand, compress)

        for index, times in reversed(sorted(expand.items())):

            if index < newCqt.shape[1] - 2:
                nc = newCqt[:, index:index+2].T
                repCol = scipy.interpolate.pchip_interpolate(np.linspace(0,
                                                                   nc.shape[0],
                                                                   nc.shape[0]),
                                                             nc,
                                                             np.linspace(0,
                                                                   nc.shape[0],
                                                                   nc.shape[0] +
                                                                   times))
                newCqt = np.insert(newCqt, index, repCol[1:-1, :], axis=1)
            else:
                for _ in range(times):
                    repCol = newCqt[:,index]
                    newCqt = np.insert(newCqt, index, repCol, axis=1)

        return newCqt

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
