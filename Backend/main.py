from MIDI_Retrieval_System import BootlegScore, MIDIProcessing, QueryProcessing, MusicalObjectDetection
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import process_data
import evaluation as eval
from skimage.metrics import structural_similarity as ssim
from mido import MidiFile

"""
Main function for the Image2Midi app

Params:
    img
    option (MIDI or PDF?)
Returns:
    MIDI
    PDF
"""


def find_image(img, option):
    """
    Main function for the Image2Midi app that finds the image and returns the MIDI or PDF
    Params:
        img: image file
        option: MIDI or PDF
    Returns:
        MIDI or PDF
    """

    bs_score_query = BootlegScore.build_from_img(img)

    if option == 'MIDI':
        # process midi batch
        dir_pkl = 'experiments/train/db'
        midi_files = os.listdir(dir_pkl)
        #print(f"Processing {len(midi_files)} MIDI files in {dir_pkl}")
        #print(f"midi_files: {midi_files}")
        all_similarity= []

        for midi in midi_files:
            midi_path = os.path.join(dir_pkl, midi)
            bscore_midi = BootlegScore.load_midi_bootleg(midi_path)

            D, wp = bscore_midi.align_to_query(bs_score_query)
            match_seg_time, _ = BootlegScore.get_predicted_timestamps(wp, bscore_midi.times)

            # get segments of the scores that match
            idxs1 = wp[::-1, 0]
            warped1 = bs_score_query.X[:,idxs1]
            idxs2 = wp[::-1, 1]
            warped2 = bscore_midi.X[:,idxs2]
            
            # compute similarity between the two segments
            similarity = ssim(warped1, warped2, data_range=warped2.max() - warped2.min())
            all_similarity.append((similarity, midi))
    
            #print(f"Comparing {midi} with {img}")
            #print(f"Similarity: {similarity}")
            #if match_seg_time is not None:
            #    print(f"Match found at time {match_seg_time} in {midi}")
       
        max_pair = max(all_similarity, key=lambda x: x[0])  # Find the pair with the highest similarity
        dir_midi = 'data/midi'
        midi_file = os.path.join(dir_midi, max_pair[1].replace('.pkl', '.mid'))
        print (f"Returning {midi_file}")
        return MidiFile(midi_file)


