import mido
import random
import scipy.io.wavfile
import math
import json
import matplotlib.pyplot     as plt
import numpy                 as np
#from dataManager             import *
from audioData               import *


class Instrument():
    """
    Class that stores all the midi notes information for a given instument.
    """
    def __init__(self, name=None, notes=None):
        self.name = name
        self.notes = notes

class MidiData():
    """
    Class that operates over a midi file. It extracts inmportat information
    from the midi file, later used as ground truth to train models and get
    performances.
    """

    def __init__(self, filename, generateWav=False):
        self.name = filename.split('/')[-1].split('.')[0].replace("""'""",
                                                         '').replace('.', '')
        self.path = filename.split(filename.split('/')[-1])[0]
        # print('\n\nIncluyendo midi '+ self.name)
        if not os.path.isfile(self.path + self.name + '.wav') and generateWav:
            self.generate_wav()
        self.instruments = []
        # There is no better way to list all percussive instruments....
        self.percussive_instruments = ['Closed Hi Hat','Tight Snare',
                                       'Hi Ride Cymbal', 'Drums',
                                       'STANDARD Drum*gemischt', 'Side Stick',
                                       'Loose Kick', 'Side Stick', 'Shaker',
                                       'Long Guiro', 'Short Guiro', 'Lo Conga',
                                       'Open Hi Conga', 'drums', 'StdDrums',
                                       'SnareDrum', 'BassDrum', 'Hihats',
                                       'Toms', 'RideCymbal', 'CrashCymbal',
                                       'Rimshots', 'Hihatfills', 'Hi-hat',
                                       'Tambourine', 'Count-in', 'Tom',
                                       'CrashCymbal', 'Snarefills', 'cymbals',
                                       'snare', 'bassdrum', 'ridecymbals',
                                       'ShortGuiro', 'LongGuiro', 'LoConga',
                                       'LooseKick', 'OpenHiConga', 'SideStick',
                                       'Percussion', 'Morepercussion',
                                       'StdDrums', 'HandClaps', 'toms',
                                       'TomToms', 'openhihat1', 'KickDrum',
                                       'Rhythm', 'OpenHiHat', 'ClosedHiHat',
                                       'TightSnare', 'Std.DrumSet', 'DRUMS',
                                       'Drums2', 'RhythmTrack', 'Sandblock',
                                       'PercussionStdDrums', '',
                                       'StandardKit', 'bateria', 'Bateria',
                                       'HiRideCymbal', 'BrushDrumKit',
                                       '*DrumSetandPercussion(Devian/Chris)',
                                       'Drum1', 'DrumSet', 'REV.CYMBL',
                                       'fill', 'Drums1', 'Drums2', 'Drums3',
                                       'Drums4', 'Drums5', 'Drums6', 'Drums7',
                                       'Drums(BB)', 'dr', 'LoRideCymbal',
                                       'Standard Kit']


        self.colors = np.array([[255,0,0], [0,155,0], [0,0,255], [0,255,255],
                                [255,0,255], [0,255,255], [100,255,100],
                                [100,100,255], [255,100,100], [100,100,100],
                                [200,200,200], [200,100,200], [50,50,50],
                                [0,20, 120],[120,20,0]], dtype=np.uint8)

        self.note_names = ('A0', 'A0#', 'B0', 'C1', 'C1#', 'D1', 'D1#', 'E1',
                           'F1', 'F1#', 'G1', 'G1#', 'A1', 'A1#', 'B1', 'C2',
                           'C2#', 'D2', 'D2#', 'E2', 'F2', 'F2#', 'G2', 'G2#',
                           'A2', 'A2#', 'B2', 'C3', 'C3#', 'D3', 'D3#', 'E3',
                           'F3', 'F3#', 'G3', 'G3#', 'A3', 'A3#', 'B3', 'C4',
                           'C4#', 'D4', 'D4#', 'E4', 'F4', 'F4#', 'G4', 'G4#',
                           'A4', 'A4#', 'B4', 'C5', 'C5#', 'D5', 'D5#', 'E5',
                           'F5', 'F5#', 'G5', 'G5#', 'A5', 'A5#', 'B5', 'C6',
                           'C6#', 'D6', 'D6#', 'E6', 'F6', 'F6#', 'G6', 'G6#',
                           'A6', 'A6#', 'B6', 'C7', 'C7#', 'D7', 'D7#', 'E7',
                           'F7', 'F7#', 'G7', 'G7#', 'A7', 'A7#', 'B7', 'C7')
        self.tempos = None
        self.meter = None
        self.measures = None
        self.resolution = 120
        self.get_instruments(filename)
        # self.midi2song(filename)
        self.max_time = self.get_max_time()
        self.beats = self.get_gt_beat()
        self.get_measures()
        self.gt_bass = self.get_gt_bass()

    def generate_wav(self):
        outname = self.name + '.wav'
        os.system('fluidsynth -ni MIDIsounds/*sf2 ' + self.path +
                  self.name + '.mid -F ' + self.path + outname + ' -r 44100')

    def add_instrument(self, instrument):
        for inst in self.instruments:
            if instrument.name == inst.name:
                return None
        self.instruments.append(instrument)

    def get_max_time(self):
        max_time = 0
        for inst in self.instruments:
            for num in inst.notes.keys():
                if inst.notes[num]['end']:
                    max_time = max(max_time, max(inst.notes[num]['end']))
                    max_time = max(max_time, max(inst.notes[num]['start']))
        return max_time

    def midifile_to_dict(self, midi_path): #將midi轉為dict
        midi_path = mido.MidiFile(midi_path)
        tracks = []
        for track in midi_path.tracks:
            tracks.append([vars(msg).copy() for msg in track])

        return {
            'ticks_per_beat': midi_path.ticks_per_beat,
            'tracks': tracks,
        }

    def get_measures(self):
        measures = []
        num = self.meter[0]
        den = self.meter[1]
        for i in range(len(self.beats)):
            if not i % int(int(num)*4/int(den)):
                measures.append(self.beats[i])

        self.measures = measures

    def get_tempos(self, midi_data):
        tempos = {}
        cumulated_time = 0
        for line_dict in midi_data['tracks'][0]:
            if 'tempo' in line_dict.keys():
                cumulated_time += line_dict['time']
                #print('time: {}, {}'.format(cumulated_time,  6e7 / line_dict['tempo']))
                tempos[cumulated_time] = 6e7 / line_dict['tempo']
        #print(tempos)
        return tempos

    def get_current_tempo(self, t):
        keys = sorted(self.tempos.keys()) + [5e8]
        for key_ind in range(len(keys)-1):
            if t >= keys[key_ind] and t < keys[key_ind+1]:
                curr_tempo = self.tempos[keys[key_ind]]
                break
        return curr_tempo

    def get_instruments(self, midi_path):
        midi_data = self.midifile_to_dict(midi_path)
        self.tempos = self.get_tempos(midi_data)
        #self.tempo = 6e7 / midi_data['tracks'][0][0]['tempo']
        self.meter = [str(midi_data['tracks'][0][1]['numerator']),
                      str(midi_data['tracks'][0][1]['denominator'])]
        self.resolution = midi_data['ticks_per_beat'] / 4


        for track in midi_data['tracks'][1:]:
            active = False
            notes = dict()
            for midi_note in range(1, 130):
                notes[midi_note] = {'start': list(), 'end': list()}
            name = track[0]['name'].replace(' ','').replace('í', 'i')
            t = 0
            t_seconds = 0
            for event in track:
                if 'note' in event.keys():
                    t += event['time']
                    curr_tempo = self.get_current_tempo(t)
                    #print('t: {}, curr_tempo: {}'.format(t, curr_tempo))
                    t_seconds += event['time'] /(4 * self.resolution * curr_tempo) * 60
                    note = event['note']
                    #print(note)
                    if 'note_on' in event['type']:
                        active = True
                        notes[note]['start'].append(t_seconds)
                    elif 'note_off' in event['type']:
                        notes[note]['end'].append(t_seconds)
            if active:
                # if (name not in self.percussive_instruments):
                #     print('name: {}'.format(name))
                self.add_instrument(Instrument(name=name, notes=notes))



    def get_gt_bass(self, low_limit=80): # G3!
        bass_gt = []
        for inst in self.instruments:
            if ((inst.name not in self.percussive_instruments) and
               ('Bater' not in inst.name) and ('bass' in inst.name or
               'Bass' in inst.name or 'bajo' in inst.name or 'Bajo' in
               inst.name)):
                for note in inst.notes.keys():
                    if int(note) <= low_limit:
                        for i in inst.notes[note]['start']:
                          #  print('i: {}, key: {}, curr.tempo: {}'.format(i, keys[key_ind], curr_tempo))
                            bass_gt.append(i)
        bass_gt = np.sort(np.array(list(set(bass_gt))))
        bass_gt = list(bass_gt)
        i = 0
        min_dist = 1e-1
        while i < len(bass_gt)-1:
            if abs(bass_gt[i+1] - bass_gt[i]) < min_dist:
                bass_gt.pop(i)
                i -= 1
            i += 1

        return np.sort(np.array(bass_gt))

    def get_gt_beat(self):
        curr_time = 0
        beats = [0]
        allKeys = list(self.tempos.keys())
        tempos = list(self.tempos.values())
        for i in range(len(allKeys) - 1):
            key = allKeys[i]
            key = int(key/(self.resolution*4))
            next_key = allKeys[i+1]
            next_key = int(next_key/(self.resolution*4))
            curr_tempo = tempos[i]
            for j in range(next_key-key):
                #print('curr_tempo es {} y estamos en beat {}/{}'.format(curr_tempo, j,next_key-key ))
                curr_time += 60/curr_tempo
                beats.append(curr_time)
        curr_tempo = tempos[-1]
       # print('curr_time: {} y max time: {}'.format(curr_time,self.max_time))
        while curr_time < self.max_time:
            curr_time += 60/curr_tempo
            beats.append(curr_time)
        return np.array(beats)
        #return np.arange(0, self.max_time / self.resolution /(self.tempo/60),
         #                60/self.tempo)



class JsonData():

    def __init__(self, path):
        file = open(path, "r", encoding="utf-8")
        self.info = json.load(file)
        self.startMeasure = int(self.info['startMeasure'])
        self.name = self.info['name']
        self.chords = None
        self.form = None
        self.chorusMeasures = None
        self.get_chords()

    def get_chords(self):
        allChords = self.info['chords']
        self.form = list(allChords.keys())
        self.chords = ''.join(list(allChords.values()))
        self.chorusMeasures = self.chords.count('|')


class InfoData(MidiData, JsonData):
    def __init__(self, songName):
        JsonData.__init__(self, songName + '.json')
        MidiData.__init__(self, songName + '_StudioOne.mid')
        self.choruses = self.get_choruses()
        self.dbName = None

    def chorus_boudaries(self):
        num, den = self.meter
        curr_beat = int((self.startMeasure-1) * int(num))
        boundaries = [self.beats[curr_beat]]
        while self.beats[curr_beat] < self.beats[-1]:
            if self.beats[curr_beat] > self.beats[-6]:
                break
            #print('condition: self.beats[curr_beat] {} < np.max(self.beats[:-8] {})'.format(self.beats[curr_beat], self.beats[-8]))
            curr_beat += int(num) * self.chorusMeasures
            #print('curr_beat {}, len(beats) {}'.format(curr_beat, len(self.beats)))
            if curr_beat == len(self.beats):
                boundaries.append(2 * self.beats[-1] - self.beats[-2])
                break
            boundaries.append(self.beats[curr_beat])
        return boundaries

    def get_choruses(self):
        choruses = {}
        boundaries = self.chorus_boudaries()
        for i in range(len(boundaries)-1):
            start = boundaries[i]
            end = boundaries[i+1]
            curr_beats = list(filter(lambda x: x >= start and x < end, self.beats))
            choruses[str(i)] = {'beats': curr_beats, 'start': start, 'end': end}
        return choruses
