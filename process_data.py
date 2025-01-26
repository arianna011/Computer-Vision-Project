from MIDI_Retrieval_System import BootlegScore
import time
import os
import multiprocessing
import logging

"""
Functions to process the query images in the dataset
"""

# standard directories for processing
query_list = 'cfg_files/query.train.list' # list of query images
midi_bs_dir = 'experiments/train/db' # directory containing midi bootleg scores
out_dir = 'experiments/train/hyp' # where to save hypothesis output files


def process_query(img_file: str, midi_bscore_pkl: str, out_file: str = None) -> tuple[float, float]:
    """
    Generate the bootleg score corresponding to the input query image file,
    align it to the MIDI bootleg and save the timestamps of the matched segment on a file

    Params:
        img_file (str): path to the input image
        midi_bscore_pkl (str): path to the corresponding MIDI pickle file
        out_file (str): path of the file where to store the resulting processing information

    Returns:
         (tuple[float, float]): predicted start and end timestamps in seconds of the segment in the MIDI file matched to the query
    """

    print(f"Processing {img_file}")
    profile_start = time.time()

    bscore_query, _, _ = BootlegScore.build_from_img(img_file)
    if bscore_query is None:
        save_to_file(out_file, img_file, (0,0), time.time() - profile_start)
        return (0,0)
    

    bscore_midi = BootlegScore.load_midi_bootleg(midi_bscore_pkl)
    D, wp = bscore_midi.align_to_query(bscore_query)
    match_seg_time, _ = BootlegScore.get_predicted_timestamps(wp, bscore_midi.times)
        
    # profile & save to file
    profile_end = time.time()
    profile_dur = profile_end - profile_start
    save_to_file(out_file, img_file, match_seg_time, profile_dur)
        
    return match_seg_time 


def save_to_file(out_file: str, img_file: str, segment: tuple[float, float], dur: float):
    """
    Save in a file the information related to processing a query image (id, timestamps of matched segment, processing time)

    Params:
        out_file (str): path of the file to write on
        img_file (str): path of the query image
        segment (tuple[float, float]): timestamps for start and end of matched segment in seconds
        dur (float): processing time
    """
    if out_file:
        with open(out_file, 'w') as f:
            query = os.path.splitext(os.path.basename(img_file))[0]
            out_str = "{},{:.2f},{:.2f},{:.2f}\n".format(query, segment[0], segment[1], dur)
            f.write(out_str)


def process_query_wrapper(query_file: str, midi_dir: str, out_dir: str) -> tuple[float, float]:
    """
    Wraps the query processing for running multiple jobs in parallel

    Params:
        query_file (str): path to the input query image (in the format "pN_qM.jpg", where N identifies the piece and M the query related to that piece)
        midi_dir (str): path to the directory containing MIDI bootleg .pkl files
        out_dir (str): path to the directory where to save processing results
    
    Returns:
        (tuple[float, float]): predicted start and end timestamps in seconds of the segment in the MIDI file matched to the query
    """
    basename = os.path.splitext(os.path.basename(query_file))[0] # e.g. p1_q1
    hyp_outfile = "{}/{}.hyp".format(out_dir, basename)
    piece = basename.split('_')[0]
    midi_bootleg_file = "{}/{}.pkl".format(midi_dir, piece)
    return process_query(query_file, midi_bootleg_file, hyp_outfile)


def process_all_queries(query_list: str = query_list, midi_bs_dir: str = midi_bs_dir, out_dir: str = out_dir) -> list[tuple[float, float]]:
    """
    Process all the query images specified in the list in input

    Params:
        query_list (str): path to the file containing the list of file paths of query images to process
        midi_bs_dir (str): path to the directory containing MIDI bootleg .pkl files
        out_dir (str): path to the directory where to store processing results

    Returns:
        (list[tuple[float, float]]): list of predicted start and end timestamps in seconds of the segments in the MIDI files matched to the corresponding queries
    """

    logging.basicConfig(level=logging.INFO)

    # prep output directory
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # number of cores to use
    n_cores = multiprocessing.cpu_count()

    # prep inputs for parallelization
    inputs = []
    with open(query_list, 'r') as f:
        for line in f:
            inputs.append((line.rstrip(), midi_bs_dir, out_dir))

    # process queries in parallel
    pool = multiprocessing.Pool(processes=n_cores)
    outputs = list(pool.starmap(process_query_wrapper, inputs)) # apply the process_query_wrapper function to each tuple in the inputs list
    pool.close()
    pool.join()

    return outputs

