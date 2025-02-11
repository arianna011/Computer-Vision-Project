import PIL.Image
from MIDI_Retrieval_System import BootlegScore, MIDIProcessing, QueryProcessing, MusicalObjectDetection
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import process_data
import evaluation as eval
from skimage.metrics import structural_similarity as ssim
from mido import MidiFile
import PIL
from multiprocessing import Pool


def main(img: PIL.Image.Image | str, option: str) -> tuple[MidiFile, tuple[float, float]]:
    """
    Main function for the Image2Midi app that, given a query image,
    based on the option input:
        - if option = "MIDI", finds and returns the matching MIDI file and the corresponding time interval
        - if option = "PDF", returns the matching PDF file
    Params:
        img (PIL.Image or str): image file or path to the query image
        option (str): "MIDI" or "PDF"
    Returns:
        (str): MIDI or PDF file path
        (tuple[float, float]): predicted timestamps in seconds of the segment of the MIDI that matches the query (only for MIDI option)
    """
    if option == "MIDI":
        return find_image(img)
    elif option == "PDF":
         return find_pdf(img)
    else:
        print("Error: wrong option input ", option)
     

def find_image(img: PIL.Image.Image | str):
    """
    Function for the Image2Midi app that, given a query image, finds and returns the matching MIDI file and the corresponding time interval
    Params:
        img (PIL.Image or str): image file or path to the image
    Returns:
        (str): file MIDI path
        (tuple[float, float]): predicted timestamps in seconds of the segment of the MIDI that matches the query
    """

    bs_score_query = BootlegScore.build_from_img(img)

    # process midi batch
    dir_midi = 'data/midi'
    dir_pkl_train = 'experiments/train/db'
    dir_pkl_test = 'experiments/test/db'
    midi_files_train = os.listdir(dir_pkl_train)
    midi_files_test = os.listdir(dir_pkl_test)

    all_similarity_train = []
    all_similarity_test = []

    with Pool() as pool:
            all_similarity_train = pool.starmap(_process_midi, [(midi, bs_score_query, dir_pkl_train) for midi in midi_files_train])
            all_similarity_test = pool.starmap(_process_midi, [(midi, bs_score_query, dir_pkl_test) for midi in midi_files_test])

    # take the MIDI with the minimum distance
    min_pair_train = min(all_similarity_train, key=lambda x: x[0])  
    min_pair_test = min(all_similarity_test, key=lambda x: x[0])
    if min_pair_train < min_pair_test:
         min_pair = min_pair_train
    else:
         min_pair = min_pair_test      
    midi_file = os.path.join(dir_midi, min_pair[1].replace('.pkl', '.mid'))
    interval = min_pair[2]
    return midi_file, interval


def find_pdf(img: PIL.Image.Image | str):
    """
    Function for the Image2Midi app that, given a query image, finds and returns the entire PDF file of the corresponding sheet music
    Params:
        img (PIL.Image or str): image file or path to the image
    Returns:
        (str): PDF file path
    """
   
    dir_pdf = 'data/pdfs'
    dir_pkl = 'experiments/train/pdf'
    pdf_files = os.listdir(dir_pkl)
    bscore_query = BootlegScore.build_from_img(img)

    all_similarity= []

    with Pool() as pool:
            all_similarity = pool.starmap(_process_pdf, [(pdf, bscore_query, dir_pkl) for pdf in pdf_files])

    # take the PDF with the minimum distance
    min_pair = min(all_similarity, key=lambda x: x[0])    
    pdf_file = os.path.join(dir_pdf, f"{min_pair[1].split('.')[0]}.pdf")

    return pdf_file, None



def _process_pdf(pdf, bscore_query, dir_pkl):
    """
    Loads the bootleg score of a PDF file and aligns it with the query image bootleg score to find the similarity score between the two
    Params:
        pdf (str): PDF file 
        bscore_query (BootlegScore): bootleg score of the query image
        dir_pkl (str): directory of the pickle files
    Returns:
        (float): cost of the alignment between the query image and the PDF file
        (str): PDF file 
    """
    pdf_path = os.path.join(dir_pkl, pdf)
    # load pdf bootleg score
    bscore_pdf = BootlegScore.load_pdf_bootleg(pdf_path)
    # align bootleg scores
    if bscore_query is None:
        end_cost = 1
    else:
        D, wp, end_cost = bscore_query.align_to_pdf(bscore_pdf)
    return end_cost, pdf



def _process_midi(midi, bscore_query, dir_pkl):
    """
    Loads the bootleg score of a MIDI file and aligns it with the query image bootleg score to find the similarity score between the two
    Params:
        midi (str): MIDI file 
        bscore_query (BootlegScore): bootleg score of the query image
        dir_pkl (str): directory of the pickle files
    Returns:
        (float): cost of the alignment between the query image and the PDF file 
        (str): MIDI file path
        (tuple[float, float]): predicted timestamps in seconds of the segment of the MIDI that matches the query
    """
    midi_path = os.path.join(dir_pkl, midi)
    # load midi bootleg score
    bscore_midi = BootlegScore.load_midi_bootleg(midi_path)
    num_query_notes = len(np.sum(bscore_query.X, axis=0))
    # align bootleg scores
    D, wp, end_cost = bscore_midi.align_to_query(bscore_query)
    match_seg_time, _ = BootlegScore.get_predicted_timestamps(wp, bscore_midi.times)  

    return end_cost, midi, match_seg_time


