from enum import pickle_by_global_name
import os
import pickle
from typing import no_type_check_decorator

from mido import MidiFile, tick2second
from numpy.typing import ArrayLike
from pretty_midi import PrettyMIDI
import numpy as np

import matplotlib.pyplot as plt


# -- visualization

def show_img(img: ArrayLike, size: tuple[int, int] = (6, 6)):
    plt.figure(figsize=size)
    plt.imshow(img, cmap='gray', origin='lower')
    plt.show()


def visualize_bootlegscore(score: np.ndarray, lines):
    show_img(1 - score, size=(10, 10))

    for l in range(1, score.shape[0], 2):
        plt.axhline(l, c='b')
    
    for l in lines:
        plt.axhline(l, c='r')


# -- notes management

def get_noteEvents(midifile, quant: int = 10) -> tuple[list[tuple[int, int, int]], dict]:
    """
    Given a midi file, return a list of (t_tick, t_sec, notes) tuples for simultaneous note events.
    """

    # 1. get note onset infos

    mid = MidiFile(midifile)
    note_events = []
    check_duplicates = {}

    for _, track in enumerate(mid.tracks):
        t = 0
        for msg in track:
            t += msg.time # how many ticks since the last event
            
            if msg.type == 'note_on' and msg.velocity > 0:
                key = '{}, {}'.format(t, msg.note)
                if key not in check_duplicates:
                    note_events.append((t, msg.note))
                    check_duplicates[key] = 0

    note_events = sorted(note_events)
    pm = PrettyMIDI(midifile)
    note_onset = [(t_ticks, pm.tick_to_time(t_ticks), note) for (t_ticks, note) in note_events]


    # 2. collapse simultaneous notes

    d = {}
    ticks_quant = [n[0]//quant for n in note_onset]
    for n, t_quant in zip(note_onset, ticks_quant):
        if t_quant not in d:
            d[t_quant] = {}
            d[t_quant]['ticks'] = []
            d[t_quant]['secs'] = []
            d[t_quant]['notes'] = []
        d[t_quant]['ticks'].append(n[0])
        d[t_quant]['secs'].append(n[1])
        d[t_quant]['notes'].append(n[2])

    result = [(d[key]['ticks'][0], d[key]['secs'][0], d[key]['notes']) for key in sorted(d.keys())]

    return result, d


def add_octave_changes(r: dict, l: dict) -> tuple[dict, dict]:
    
    # add octaves in treble clef for G5 and above
    for midinum in r:
        if midinum >= 79:
            to_add = []
            for staffpos in r[midinum]:
                to_add.append(staffpos - 7) # 7 staff positions = 1 octave
            r[midinum].extend(to_add)

    # add octaves in bass clef for F2 and below
    for midinum in l:
        if midinum <= 41:
            to_add = []
            for staffpos in l[midinum]:
                to_add.append(staffpos + 7)
            l[midinum].extend(to_add)

    return r, l


def add_clef_changes(r: dict, l: dict) -> tuple[dict, dict]:

    # clef changes in rh
    for midinum in range(36, 65): # C2 to E4
        if midinum not in r:
            r[midinum] = []
        for staffpos in l[midinum]:
            r[midinum].append(staffpos - 6) # shift between L and R staves 

    # clef changes in lh
    for midinum in range(57, 85): # A3 to C6
        if midinum not in l:
            l[midinum] = []
        for staffpos in r[midinum]:
            l[midinum].append(staffpos + 6)

    return r, l


def getNoteheadPlacementMappingLH() -> dict:
    d = {}
    # e.g. d[23] = [1,2] indicates that B0 could appear as a B or a C-flat, which means
    # that the notehead could be located at positions 1 or 2
    d[21] = [0] # A0 (position 0)
    d[22] = [0,1]
    d[23] = [1,2] # B0
    d[24] = [1,2] # C1
    d[25] = [2,3]
    d[26] = [3] # D1
    d[27] = [3,4]
    d[28] = [4,5] # E1
    d[29] = [4,5] # F1
    d[30] = [5,6]
    d[31] = [6] # G1
    d[32] = [6,7] 
    d[33] = [7] # A1
    d[34] = [7,8]
    d[35] = [8,9] # B1
    d[36] = [8,9] # C2
    d[37] = [9,10] 
    d[38] = [10] # D2
    d[39] = [10,11] 
    d[40] = [11,12] # E2
    d[41] = [11,12] # F2
    d[42] = [12,13] 
    d[43] = [13] # G2
    d[44] = [13,14] 
    d[45] = [14] # A2
    d[46] = [14,15] 
    d[47] = [15,16] # B2
    d[48] = [15,16] # C3
    d[49] = [16,17] 
    d[50] = [17] # D3
    d[51] = [17,18] 
    d[52] = [18,19] # E3
    d[53] = [18,19] # F3
    d[54] = [19,20] 
    d[55] = [20] # G3
    d[56] = [20,21] 
    d[57] = [21] # A3
    d[58] = [21,22] 
    d[59] = [22,23] # B3
    d[60] = [22,23] # C4
    d[61] = [23,24] 
    d[62] = [24] # D4
    d[63] = [24,25] 
    d[64] = [25,26] # E4
    d[65] = [25,26] # F4
    d[66] = [26,27] 
    d[67] = [27] # G4
    return d


def getNoteheadPlacementMappingRH() -> dict:
    d = {}
    # e.g. d[52] = [0,1] indicates that E3 could appear as an E or an F-flat, which means
    # that the notehead could be located at positions 0 or 1
    d[52] = [0,1] # E3 (position 0)
    d[53] = [0,1] # F3
    d[54] = [1,2]
    d[55] = [2] # G3
    d[56] = [2,3]
    d[57] = [3] # A3
    d[58] = [3,4]
    d[59] = [4,5] # B3
    d[60] = [4,5] # C4
    d[61] = [5,6]
    d[62] = [6] # D4
    d[63] = [6,7]
    d[64] = [7,8] # E4
    d[65] = [7,8] # F4
    d[66] = [8,9]
    d[67] = [9] # G4
    d[68] = [9,10]
    d[69] = [10] # A4
    d[70] = [10,11]
    d[71] = [11,12] # B4
    d[72] = [11,12] # C5
    d[73] = [12,13]
    d[74] = [13] # D5
    d[75] = [13,14]
    d[76] = [14,15] # E5
    d[77] = [14,15] # F5
    d[78] = [15,16]
    d[79] = [16] # G5
    d[80] = [16,17]
    d[81] = [17] # A5
    d[82] = [17,18] 
    d[83] = [18,19] # B5
    d[84] = [18,19] # C6
    d[85] = [19,20]
    d[86] = [20] # D6
    d[87] = [20,21]
    d[88] = [21,22] # E6
    d[89] = [21,22] # F6
    d[90] = [22,23]
    d[91] = [23] # G6
    d[92] = [23,24] 
    d[93] = [24] # A6
    d[94] = [24,25]
    d[95] = [25,26] # B6
    d[96] = [25,26] # C7
    d[97] = [26,27]
    d[98] = [27] # D7
    d[99] = [27,28] 
    d[100] = [28,29] # E7
    d[101] = [28,29] # F7
    d[102] = [29,30]
    d[103] = [30] # G7
    d[104] = [30,31]    
    d[105] = [31] # A7
    d[106] = [31,32]
    d[107] = [32,33] # B7
    d[108] = [32,33] # C8
    return d

def get_notehead_placement_mapping() -> tuple[dict, dict]:
    r = getNoteheadPlacementMappingRH()
    l = getNoteheadPlacementMappingLH()
    
    # r, l = add_octave_changes(r, l) # include octave markings
    # r, l = add_clef_changes(r, l) # include different clefs
    return r, l


def get_notehead_placement(midinum, midi2loc, dim: int) -> np.ndarray:
    r = np.zeros((dim, 1))

    if midinum in midi2loc:
        for idx in midi2loc[midinum]:
            r[idx, 0] = 1
    
    return r


# -- bootleg score generation

def generate_bootlegScore(note_events, repeat_notes: int = 1, filler: int = 1) -> tuple[np.ndarray, list[tuple], list[int], list[int], tuple[np.ndarray, list[int]], tuple[np.ndarray, list[int]]] :
    rh_dim = 34 # E3 to C8 (inclusive)
    lh_dim = 28 # A1 to G4 (inclusive)

    rh = [] # list of arrays of size rh_dim
    lh = [] # list of arrays of size lh_dim

    numNotes = [] # number of simultaneous notes
    times = [] # list of (tsec, ttick) tuples indicating the time in ticks and seconds
    mapR, mapL = get_notehead_placement_mapping() # maps midi numbers to locations on right and left hand staves

    for i, (ttick, tsec, notes) in enumerate(note_events):
        
        # insert empty filler columns between note events
        if i > 0:
            for _ in range(filler):
                rh.append(np.zeros((rh_dim,1)))
                lh.append(np.zeros((lh_dim,1)))
                numNotes.append(0)
            # get corresponding times using linear interpolation
            interp_ticks = np.interp(np.arange(1, filler+1), [0, filler+1], [note_events[i-1][0], ttick])
            interp_secs = np.interp(np.arange(1, filler+1), [0, filler+1], [note_events[i-1][1], tsec])
            for tup in zip(interp_secs, interp_ticks):
                times.append((tup[0], tup[1]))

        # insert note events columns
        rhvec = np.zeros((rh_dim, 1))
        lhvec = np.zeros((lh_dim, 1))

        for midinum in notes:
            rhvec += get_notehead_placement(midinum, mapR, rh_dim)
            lhvec += get_notehead_placement(midinum, mapL, lh_dim)

        for _ in range(repeat_notes):
            rh.append(rhvec)
            lh.append(lhvec)
            numNotes.append(len(notes))
            times.append((tsec, ttick))

    rh = np.clip(np.squeeze(np.array(rh)).T, 0, 1) # clip in case e.g. E and F played simultaneously
    lh = np.clip(np.squeeze(np.array(lh)).T, 0, 1) 

    both = np.vstack((lh, rh))

    staffLinesRH = [7,9,11,13,15]
    staffLinesLH = [13,15,17,19,21]
    staffLinesBoth = [13,15,17,19,21,35,37,39,41,43]

    return both, times, numNotes, staffLinesBoth, (rh, staffLinesRH), (lh, staffLinesLH)


# -- midi processing

def process_midi(midifile, outfile: str):

    # paramenters
    time_quant_factor = 10
    bootleg_repeat_notes = 2
    bootleg_filler = 1
    # end

    print("huehuehue processing!!! {}".format(midifile))

    note_events, _ = get_noteEvents(midifile, time_quant_factor)
    bscore, times, num_notes, stafflines, _, _ = generate_bootlegScore(note_events, bootleg_repeat_notes, bootleg_filler)


    # saving to file
    d = {
            'bscore': bscore,
            'times': times,
            'num_notes': num_notes,
            'stafflines': stafflines,
            'note_events': note_events
        }

    with open(outfile, 'wb') as file:
        pickle.dump(d, file)


def process_midi_batch(file_list: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    with open(file_list, 'r') as file:
        for f in file:
            f = f.rstrip()
            basename = os.path.splitext(os.path.basename(f))[0]
            outfile = "{}.wooper".format(os.path.join(outdir, basename))
            process_midi(f, outfile)


# -- testing

def test_midi_processing(midifile: str = 'midi.test.list'):
    file_list = os.path.join('cfg_files', midifile)
    basedir = os.path.join('experiments', 'train')
    outdir = os.path.join(basedir, 'db')
    process_midi_batch(file_list, outdir)
    

def test_bootleg_score(track: str = '91'):
    midifile = os.path.join('data', 'midi', 'p{}.mid'.format(track)) 

    note_events, _ = get_noteEvents(midifile)
    bscore, _, _, stafflines, _, _ = generate_bootlegScore(note_events, 2, 2)

    visualize_bootlegscore(bscore[:, 0:140], stafflines)
