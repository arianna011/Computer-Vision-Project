import PIL.Image
from .MIDI_Retrieval_System import BootlegScore
import os
from . import evaluation as eval
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
        (MidiFile): file MIDI
        (tuple[float, float]): predicted timestamps in seconds of the segment of the MIDI that matches the query
        pdf
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
        (MidiFile): file MIDI
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

    # take the MIDI with the maximum similarity
    max_pair_train = max(all_similarity_train, key=lambda x: x[0])  
    max_pair_test = max(all_similarity_test, key=lambda x: x[0])
    if max_pair_train > max_pair_test:
         max_pair = max_pair_train
    else:
         max_pair = max_pair_test      
    midi_file = os.path.join(dir_midi, max_pair[1].replace('.pkl', '.mid'))
    interval = max_pair[2]
    print(f"Query {img}: Returning {midi_file}, interval ({interval[0]} s, {interval[1]} s)")
    return MidiFile(midi_file), interval


def find_pdf(img: PIL.Image.Image | str):
    """
    Function for the Image2Midi app that, given a query image, finds and returns the entire PDF file of the corresponding sheet music
    Params:
        img (PIL.Image or str): image file or path to the image
    Returns:
        (): PDF file
    """
   
    dir_pdf = 'data/pdfs'
    dir_pkl = 'experiments/train/pdf'
    pdf_files = os.listdir(dir_pkl)
    bscore_query = BootlegScore.build_from_img(img)

    all_similarity= []

    with Pool() as pool:
            all_similarity = pool.starmap(_process_pdf, [(pdf, bscore_query, dir_pkl) for pdf in pdf_files])

    # take the PDF with the maximum similarity
    max_pair = max(all_similarity, key=lambda x: x[0])    
    pdf_file = os.path.join(dir_pdf, f"{max_pair[1].split('_')[0]}.pdf")
    print(f"Query {img}: Returning {pdf_file}")
    return pdf_file



def _process_pdf(pdf, bscore_query, dir_pkl):
    pdf_path = os.path.join(dir_pkl, pdf)
    bscore_pdf = BootlegScore.load_pdf_bootleg(pdf_path)
    D, wp = bscore_query.align_to_pdf(bscore_pdf)
    return 1 / (1 + D[-1, -1]), pdf



def _process_midi(midi, bscore_query, dir_pkl):
    midi_path = os.path.join(dir_pkl, midi)
    bscore_midi = BootlegScore.load_midi_bootleg(midi_path)
    D, wp = bscore_midi.align_to_query(bscore_query)
    match_seg_time, _ = BootlegScore.get_predicted_timestamps(wp, bscore_midi.times)   
    return _compute_similarity(wp, bscore_query, bscore_midi), midi, match_seg_time

def _compute_similarity(wp, bscore_query, bscore_midi):
    idxs1 = wp[::-1, 0]
    warped1 = bscore_query.X[:, idxs1]
    idxs2 = wp[::-1, 1]
    warped2 = bscore_midi.X[:, idxs2]

    warped1 = (warped1 - warped1.min()) / (warped1.max() - warped1.min())
    warped2 = (warped2 - warped2.min()) / (warped2.max() - warped2.min())

    similarity = ssim(warped1, warped2, data_range = max(warped1.max(), warped2.max()) - min(warped1.min(), warped2.min()))
    return similarity


# def _compute_similarity(D):
#     return np.exp(-D[-1, -1])
