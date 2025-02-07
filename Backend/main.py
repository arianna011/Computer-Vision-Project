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

        all_similarity= []

        for midi in midi_files:
            # path from which we load the midi file 
            midi_path = os.path.join(dir_pkl, midi)
            bscore_midi = BootlegScore.load_midi_bootleg(midi_path)
            # align the query to the midi with dynamic time warping
            D, wp = bscore_midi.align_to_query(bs_score_query)
            match_seg_time, _ = BootlegScore.get_predicted_timestamps(wp, bscore_midi.times)

            # get segments of the scores that match
            idxs1 = wp[::-1, 0]
            warped1 = bs_score_query.X[:,idxs1]
            idxs2 = wp[::-1, 1]
            warped2 = bscore_midi.X[:,idxs2]
            
            # compute similarity between the two segments
            # TODO: try using DTW data instead of SSIM
            #similarity = 1 / (1 + D[-1, -1])
            similarity = ssim(warped1, warped2, data_range=warped2.max() - warped2.min())
            all_similarity.append((similarity, midi))


            # TODO: this is for debugging
            print(f"Comparing {midi} with {img}")
            print(f"Similarity: {similarity}")
       
        max_pair = max(all_similarity, key=lambda x: x[0])  # Find the pair with the highest similarity
        dir_midi = 'data/midi'
        midi_file = os.path.join(dir_midi, max_pair[1].replace('.pkl', '.mid'))
        print (f"Returning {midi_file}")
        return MidiFile(midi_file)

    if option == 'PDF':

        dir_pkl = 'experiments/train/pdf'
        pdf_files = os.listdir(dir_pkl)

        all_similarity= []

        for pdf in pdf_files:
            # path from which we load the pdf file
            pdf_path = os.path.join(dir_pkl, pdf)
            bscore_pdf = BootlegScore.load_pdf_bootleg(pdf_path)

            #TODO: fix function align_to_pdf
            D, wp = bs_score_query.align_to_pdf(bscore_pdf)
            #match_seg_time, _ = BootlegScore.get_predicted_timestamps(wp, bscore_pdf.times)

            # get segments of the scores that match
            idxs1 = wp[::-1, 0]
            warped1 = bs_score_query.X[:,idxs1]
            idxs2 = wp[::-1, 1]
            warped2 = bscore_pdf.X[:,idxs2]
            
            # compute similarity between the two segments
            #similarity = ssim(warped1, warped2, data_range=warped2.max() - warped2.min())
            similarity = 1 / (1 + D[-1, -1])  

            all_similarity.append((similarity, pdf))
    
            print(f"Comparing {pdf} with {img}")
            print(f"Similarity: {similarity}")
            #if match_seg_time is not None:
            #   print(f"Match found at time {match_seg_time} in {pdf}")
        
        max_pair = max(all_similarity, key=lambda x: x[0])  # Find the pair with the highest similarity
        pdf_dir = 'data/pdfs'
        pdf_file = os.path.join(pdf_dir, max_pair[1].replace('.pkl', '.pdf'))
        print (f"Returning {pdf_file}")
        return pdf_file
