import numpy as np
import os
import pickle
from mido import MidiFile
from pretty_midi import PrettyMIDI
from collections import defaultdict


class MIDIProcessing:
    """
    Process a MIDI file by extracting note events and converting it to a booleg score.
    """

    # system parameters 
    time_quant_factor = 10
    bootleg_repeat_notes = 2
    bootleg_filler = 1

    # positions of staff lines for right hand and left hand staves
    staff_lines_RH = [7,9,11,13,15]
    staff_lines_LH = [13,15,17,19,21]
    staff_lines_both = [13,15,17,19,21,35,37,39,41,43]

    def __init__(self, midifile: str):
        """
        Initialize the MIDIProcessing with the path of the MIDI file to process.
        """
        self.midifile = midifile

    def process(self, outfile: str):
        """
        Convert the MIDI file to a bootleg score and save it as a serialized (with pickle) object in outfile.
        """
        print(f"Processing {self.midifile}")

        note_events, _ = self.get_note_events(MIDIProcessing.time_quant_factor)
        bscore, times, num_notes, _, _ = self.generate_bootleg_score(note_events, MIDIProcessing.bootleg_repeat_notes, MIDIProcessing.bootleg_filler)
        
        # saving to file
        d = {
                'bscore': bscore,
                'times': times,
                'num_notes': num_notes,
                'stafflines': MIDIProcessing.staff_lines_both,
                'note_events': note_events
            }

        with open(outfile, 'wb') as file:
            pickle.dump(d, file)

    @staticmethod
    def process_midi_batch(file_list: str, outdir: str):
        """
        Process the batch of MIDI files specified in the list contained in file_list and store the results in outdir.
        """
        os.makedirs(outdir, exist_ok=True)

        with open(file_list, 'r') as file:
            for f in file:
                f = f.rstrip()
                basename = os.path.splitext(os.path.basename(f))[0]
                outfile = f"{outdir}/{basename}.pkl"
                MIDIProcessing(f).process(outfile)


    def get_note_events(self, quant: int = 10):
        """
        Return a list of (t_tick, t_sec, notes) tuples 
        representing simultaneous note events from the MIDI file.
        Events are considered simultaneous if they fall into the same ticks interval 
        after quantizing their absolute tick timestamp.

        (Ticks are a fine-grained relative time unit in MIDI files: each event
         - e.g. note-on, note-off - specifies how many ticks have passed since the previous event)

        Parameters:
            quant: Quantization factor in ticks.

        Output:
            result (list): List of (tick, time, notes) tuples.
            d (dict): Dictionary for debugging, mapping quantized times to notes.
        """

        # Load MIDI file
        mid = MidiFile(self.midifile)
        note_events = []

        # Extract note onset information
        for track in mid.tracks: # a MIDI file can have multiple parallel tracks
            t = 0   # absolute tick timestamp
            for msg in track:
                t += msg.time  # accumulate ticks since last event
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_events.append((t, msg.note)) # msg.note is an int representing the MIDI note number (pitch in the range 0 to 127)
    
        # Remove duplicates and sort note events from all tracks by tick time
        sorted(set(note_events))

        # Convert ticks to seconds using PrettyMIDI
        pm = PrettyMIDI(self.midifile)
        note_onsets = [(t_ticks, pm.tick_to_time(t_ticks), note) for (t_ticks, note) in note_events]

        # Group simultaneous notes based on quantized absolute tick time
        d = defaultdict(lambda: {'ticks': [], 'secs': [], 'notes': []})
        for n in note_onsets:
            t_quant = n[0] // quant  # Quantized time units (ticks)
            d[t_quant]['ticks'].append(n[0])
            d[t_quant]['secs'].append(n[1])
            d[t_quant]['notes'].append(n[2])

        # Create final list with first tick/sec of each group, sorted by quantized ticks
        result = [(d[key]['ticks'][0], d[key]['secs'][0], d[key]['notes']) for key in sorted(d.keys())]

        return result, d  # Return d for debugging

    @staticmethod
    def get_notehead_placement(midinum: int, midi2loc: dict, dim: int):
        """
        Given:
        - a MIDI note number 
        - a dictionary mapping MIDI note numbers to possible locations in a musical staff
        - a dimension for the output column array (number of rows)
        Returns:
        an array with shape (dim x 1) with 1s in the positions indicated by the input dictionary for the MIDI note number
        """
        r = np.zeros((dim, 1))
        if midinum in midi2loc:
            for idx in midi2loc[midinum]:
                r[idx,0] = 1
        return r

    @staticmethod
    def get_notehead_placement_mapping():
        """
        Get the mappings from MIDI note numbers to possible note positions 
        in the staves for the right and the left hand.
        """
        r = MIDIProcessing.get_notehead_placement_mapping_RH()
        l = MIDIProcessing.get_notehead_placement_mapping_LH()
        return r, l

    @staticmethod
    def get_notehead_placement_mapping_RH():
        """
        Get a dictionary where:
        - keys are MIDI note numbers
        - values are lists of all the possible posititions in the right-hand staff where the key note could appear
        """
        d = {}
        # e.g. d[52] = [0,1] indicates that E3 could appear as an E or an F-flat
        # which means that the notehead could be located at positions 0 or 1
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

    @staticmethod
    def get_notehead_placement_mapping_LH():
        """
        Get a dictionary where:
        - keys are MIDI note numbers
        - values are lists of all the possible posititions in the left-hand staff where the key note could appear
        """
        d = {}
        # e.g. d[23] = [1,2] indicates that B0 could appear as a B or a C-flat
        # which means that the notehead could be located at positions 1 or 2
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
    
    def generate_bootleg_score(self, note_events: list[tuple[int, int, list[int]]], repeat_notes: int = 1, 
                               filler: int = 1) -> tuple[np.ndarray, list[tuple[int, int]], list[int], tuple[np.ndarray, list[int]], tuple[np.ndarray, list[int]]]:
        """
        Generate a bootleg score as a NumPy matrix starting from simultaenous note events collected from a MIDI file.
        To improve empirical results, events are repeated and separated by empty filler columns.

        Params:
            note_events (list[tuple[int, int, list[int]]]): list of (tick, time, notes) tuples for each note event
            repeat_notes (int): how many times to repeat each note event
            filler (int): how many filler colummns to insert between note events
        Returns:
            (np.ndarray): the bootleg score for both right and left hands
            (list[tuple[int, int]]): list of (tsec, ttick) tuples indicating the time in ticks and seconds for each event
            (list[int]): list with the number of simultaneous notes in each event
            (tuple[np.ndarray, list[int]]): tuple with the bootleg score for the right hand and the right hand staff lines positions
            (tuple[np.ndarray, list[int]]): tuple with the bootleg score for the left hand and the left hand staff lines positions
        """
        rh_dim = 34 # notes included in the right hand staff: E3 to C8 (inclusive)
        lh_dim = 28 # notes included in the left hand staff: A1 to G4 (inclusive)
        rh = [] # list of arrays of size rh_dim
        lh = [] # list of arrays of size lh_dim
        num_notes = [] # array with the number of simultaneous notes in each event
        times = [] # list of (tsec, ttick) tuples indicating the time in ticks and seconds for each event
        mapR, mapL = MIDIProcessing.get_notehead_placement_mapping() # maps from MIDI note numbers to locations on right and left hand staves

        for i, (ttick, tsec, notes) in enumerate(note_events):

            # insert empty filler columns between note events
            if i > 0:
                for _ in range(filler):
                    rh.append(np.zeros((rh_dim,1)))
                    lh.append(np.zeros((lh_dim,1)))
                    num_notes.append(0)
                # get corresponding times using linear interpolation
                interp_ticks = np.interp(np.arange(1, filler+1), [0, filler+1], [note_events[i-1][0], ttick])
                interp_secs = np.interp(np.arange(1, filler+1), [0, filler+1], [note_events[i-1][1], tsec])
                for tup in zip(interp_secs, interp_ticks):
                    times.append((tup[0], tup[1]))

            # insert note events columns
            rhvec = np.zeros((rh_dim, 1))
            lhvec = np.zeros((lh_dim, 1))
            for midinum in notes:
                rhvec += MIDIProcessing.get_notehead_placement(midinum, mapR, rh_dim)
                lhvec += MIDIProcessing.get_notehead_placement(midinum, mapL, lh_dim)
            for _ in range(repeat_notes):
                rh.append(rhvec)
                lh.append(lhvec)
                num_notes.append(len(notes))
                times.append((tsec, ttick))

        rh = np.clip(np.squeeze(np.array(rh)).T, 0, 1) # clip between 0 and 1 to avoid invalid values (in case e.g. E and F are played simultaneously)
        lh = np.clip(np.squeeze(np.array(lh)).T, 0, 1)
        both = np.vstack((lh, rh)) # shape: 62 x Y where Y = number of repeated simultanoeus note events in the MIDI file + filler columns

        return both, times, num_notes, (rh, MIDIProcessing.staff_lines_RH), (lh,  MIDIProcessing.staff_lines_LH)