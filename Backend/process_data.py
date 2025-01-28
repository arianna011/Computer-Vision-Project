import MIDI_Retrieval_System.bootleg_score as bs
import time
import os
import multiprocessing
import logging
import glob
import pretty_midi
import pdf2image
import pickle

"""
Functions to process the query images in the dataset
"""

# standard directories for processing
query_list = 'cfg_files/query.train.list' # list of query images paths
midi_list = 'cfg_files/midi.train.list' # list of midi files paths
pdf_list = 'cfg_files/pdf.train.list' # list of pdf files paths
midi_bs_dir = 'experiments/train/db' # directory containing midi bootleg scores
out_dir = 'experiments/train/hyp' # where to save hypothesis output files
pdf_bs_dir = 'experiments/train/pdf' # directory containing pdf bootleg scores

score_info = 'data/score_info'
pdf_dir = 'data/pdfs' # directory containing pdf files
midi_info = 'data/midi_info'
midi_dir = 'data/midi'
query_info = 'data/query_info/query_info.csv'
mult_matches_file ='data/query_info/query.multmatches'
query_gt_file = 'data/query_info/query.gt'


def process_all_midis(file_list: str = midi_list, out_dir: str = midi_bs_dir, re_compute: bool = False):
    """
    Process the batch of MIDI files specified in the input file
    by converting them to bootleg scores and storing them in the output directory as .pkl files

    Params:
        file_list (str): path to the file containing the paths of the MIDI files to process
        out_dir (str): path of the output directory
        re_compute (bool): whether to re-compute pre-existing MIDI bootlegs
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(file_list, 'r') as file:
        for f in file:
            f = f.rstrip()
            basename = os.path.splitext(os.path.basename(f))[0]
            outfile = f"{out_dir}/{basename}.pkl"
            if (not re_compute) and (os.path.isfile(outfile)): # file already exists
                print(f'Skipping {outfile}')
            else:
                bs.MIDIProcessing(f).process(outfile)


def process_all_pdfs(file_list: str = pdf_list, out_dir: str = pdf_bs_dir, re_compute: bool = False):
    """
    Process the batch of PDF files specified in the input file
    by converting them to bootleg scores and storing them in the output directory as .pkl files

    Params:
        file_list (str): path to the file containing the paths of the PDF files to process
        out_dir (str): path of the output directory
        re_compute (bool): whether to re-compute pre-existing PDF bootlegs
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(file_list, 'r') as file:
        for f in file:
            f = f.rstrip()
            images = pdf2image.convert_from_path(f)
            for i, img in enumerate(images):
                out_file = f"{out_dir}/{os.path.splitext(os.path.basename(f))[0]}_{i}.pkl"
                
                if (not re_compute) and (os.path.isfile(out_file)): # file already exists
                    print(f'Skipping {out_file}')
                else:
                    bootleg_score_img = bs.BootlegScore.build_from_img(img)

                    print(f'Processing {out_file}')
                    if bootleg_score_img is None:
                        print(f'No bootleg score found for {out_file}')
                        continue
                    # saving to file
                    d = {
                            'bscore': bootleg_score_img,
                            'image_file': img
                        }
                    with open(out_file, 'wb') as file:
                        pickle.dump(d, file)
                
                    




def process_all_queries(query_list: str = query_list, midi_bs_dir: str = midi_bs_dir, out_dir: str = out_dir, re_compute: bool = False) -> list[tuple[float, float]]:
    """
    Process all the query images specified in the list in input

    Params:
        query_list (str): path to the file containing the list of file paths of query images to process
        midi_bs_dir (str): path to the directory containing MIDI bootleg .pkl files
        out_dir (str): path to the directory where to store processing results
        re_compute (bool): whether to overwrite already existing output files

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
            inputs.append((line.rstrip(), midi_bs_dir, out_dir, re_compute))

    # process queries in parallel
    pool = multiprocessing.Pool(processes=n_cores)
    outputs = list(pool.starmap(process_query_wrapper, inputs)) # apply the process_query_wrapper function to each tuple in the inputs list
    pool.close()
    pool.join()

    return outputs


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

    bscore_query = bs.BootlegScore.build_from_img(img_file)
    if bscore_query is None:
        save_to_file(out_file, img_file, (0,0), time.time() - profile_start)
        return (0,0)
    

    bscore_midi = bs.BootlegScore.load_midi_bootleg(midi_bscore_pkl)
    D, wp = bscore_midi.align_to_query(bscore_query)
    match_seg_time, _ = bs.BootlegScore.get_predicted_timestamps(wp, bscore_midi.times)
        
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


def process_query_wrapper(query_file: str, midi_dir: str, out_dir: str, re_compute: bool = False) -> tuple[float, float]:
    """
    Wraps the query processing for running multiple jobs in parallel

    Params:
        query_file (str): path to the input query image (in the format "pN_qM.jpg", where N identifies the piece and M the query related to that piece)
        midi_dir (str): path to the directory containing MIDI bootleg .pkl files
        out_dir (str): path to the directory where to save processing results
        re_compute (bool): whether to overwrite already existing output files
    
    Returns:
        (tuple[float, float]): predicted start and end timestamps in seconds of the segment in the MIDI file matched to the query
    """
    basename = os.path.splitext(os.path.basename(query_file))[0] # e.g. p1_q1
    hyp_outfile = "{}/{}.hyp".format(out_dir, basename)
    piece = basename.split('_')[0]
    midi_bootleg_file = "{}/{}.pkl".format(midi_dir, piece)

    if (not re_compute) and (os.path.isfile(hyp_outfile)): # file already exists
        print(f'Opening {hyp_outfile}')
        with open(hyp_outfile, 'r') as f:
            line = f.readline()  
            _, t_1, t_2, _ = line.strip().split(",")  
        return (t_1, t_2)

    return process_query(query_file, midi_bootleg_file, hyp_outfile)


def import_score_info(score_dir: str = score_info) -> dict[int, dict[int, tuple[int,int]]]:
    """
    Read score information (start measure and end measure for each line) from the files contained in the input directory

    Params:
        score_dir (str): path to the directory containing score info files

    Returns:
        (dict[int, dict[int, tuple[int,int]]]): dictonary indexed by piece numbers containing
                                                dictionaries with line numbers as keys and tuples of start measure and end measure as values
    """
    d = {}
    for csvfile in glob.glob("{}/p*.scoreinfo.csv".format(score_dir)):
        piece_str = os.path.basename(csvfile).split('.')[0]  # e.g. 'p7'
        d[piece_str] = {}
        with open(csvfile, 'r') as f:
            next(f) # skip header
            for line in f:
                parts = line.rstrip().split(',')
                line_num = int(parts[0])
                start_measure = int(parts[1])
                end_measure = int(parts[2])
                d[piece_str][line_num] = (start_measure, end_measure)
    return d


def import_midi_info(midi_info_dir: str = midi_info, midi_dir: str = midi_dir) -> dict[int, dict[int, float]]:
    """
    Read or extract MIDI information from the files contained in the input directories

    Params:
        midi_info_dir (str): path to the directory containing MIDI info files
        midi_dir (str): path to the directory containing .mid files

    Returns:
        (dict[int, dict[int, float]]): dictonary indexed by piece numbers containing
                                       dictionaries with measure numbers as keys and time duration as values
                                       (an additional entry identified by the last measure number + 1
                                        contains the total duration of the MIDI piece)
    """
    d = {}
    for csvfile in glob.glob("{}/p*_midinfo.csv".format(midi_info_dir)):
        piece_str = os.path.basename(csvfile).split('_')[0]  # e.g. 'p7'
        d[piece_str] = {}
        with open(csvfile, 'r') as f:
            for line in f:
                parts = line.rstrip().split(',')
                measure = int(parts[0])
                time = float(parts[1])
                d[piece_str][measure] = time
        
        # add an additional entry to indicate the total duration
        midfile = "{}/{}.mid".format(midi_dir, piece_str)
        mid = pretty_midi.PrettyMIDI(midfile)
        total_dur = mid.get_piano_roll().shape[1] * .01 # default sampling frequency: fs = 100
        d[piece_str][measure+1] = total_dur
                
    return d

def get_query_ground_truth(score_info: dict[int, dict[int, tuple[int,int]]], midi_info: dict[int, dict[int, float]], 
                           query_info_file: str = query_info, mult_match_file: str = mult_matches_file) -> dict[str, list[tuple[float, float, int, int, int, int]]]:
    """
    Infer ground truth timestamps for each query specified in the query info input file (containing queries id, start and end line)

    Params:
        score_info (dict[int, dict[int, tuple[int,int]]]): dictonary indexed by piece numbers containing
                                                           dictionarys with line numbers as keys and tuples of start measure and end measure as values
        midi_info (dict[int, dict[int, float]]): dictonary indexed by piece numbers containing
                                                 dictionaries with measure numbers as keys and time duration as values
                                                 (an additional entry identified by the last measure number + 1
                                                  contains the total duration of the MIDI piece)
        query_info_file (str): path to the .csv file containg queries info
        mult_match_file (str): path to the file containing information about queries with multiple matches

    Returns:
        (dict[str, list[tuple[float, float, int, int, int, int]]]): dictionary indexed by query ids, containing a list of tuples with start and end time, start and end measure, 
                                                                    start and end line of each matched segment
    """
    d = {}
    with open(query_info_file, 'r') as fin: 
        next(fin) # skip header
        for line in fin:
            # get start, end lines
            parts = line.rstrip().split(',')  # e.g. 'p1_q1,0,3'
            query_str = parts[0]
            start_line = int(parts[1])
            end_line = int(parts[2])

            print(f'Processing {query_str}')
            # infer start, end measure
            piece_str = query_str.split('_')[0]        
            start_measure = score_info[piece_str].get(start_line, score_info[piece_str][min(score_info[piece_str].keys())])[0]
            end_measure = score_info[piece_str].get(end_line, score_info[piece_str][max(score_info[piece_str].keys())])[1]

            # infer start, end time
            start_time = midi_info[piece_str][start_measure]
            end_time = midi_info[piece_str][end_measure+1] # ends on downbeat of next measure

            d[query_str] = [(start_time, end_time, start_measure, end_measure, start_line, end_line)]

    add_multiple_matches(d, mult_match_file, score_info, midi_info)         
    return d   


def add_multiple_matches(d: dict[str, list[tuple[float, float, int, int, int, int]]], mult_match_file: str, 
                         score_info: dict[int, dict[int, tuple[int,int]]], midi_info: dict[int, dict[int, float]]):
    """
    Read information about queries that match more than 1 segment of the score from the input file and add it to the input dictionary
    
    Params:
        d (dict[str, list[tuple[float, float, int, int, int, int]]]): dictionary indexed by query ids, containing a list of tuples with start and end time, 
                                                                      start and end measure, start and end line of each matched segment
        mult_match_file (str): path to the file containing information about queries with multiple matches
        score_info (dict[int, dict[int, tuple[int,int]]]): dictonary indexed by piece numbers containing
                                                           dictionarys with line numbers as keys and tuples of start measure and end measure as values
        midi_info (dict[int, dict[int, float]]): dictonary indexed by piece numbers containing
                                                 dictionaries with measure numbers as keys and time duration as values
                                                 (an additional entry identified by the last measure number + 1
                                                  contains the total duration of the MIDI piece)

    """
    # some queries match more than 1 segment of the score, these are indicated in mult_match_file
    with open(mult_match_file, 'r') as f:
        for line in f:     
            # parse line 
            parts = line.rstrip().split(',')  # e.g. 'p31_q8,L3m6,L5m1'
            query_str = parts[0]
            piece_str = query_str.split('_')[0]
            start_str = parts[1]
            end_str = parts[2]
            
            # infer start, end measure
            start_line = int(start_str.split('m')[0][1:])
            end_line = int(end_str.split('m')[0][1:])
            start_offset = int(start_str.split('m')[1])
            end_offset = int(end_str.split('m')[1])
            start_measure = score_info[piece_str][start_line][0] + start_offset - 1
            end_measure = score_info[piece_str][end_line][0] + end_offset - 1
            
            # infer start, end time
            start_time = midi_info[piece_str][start_measure]
            end_time = midi_info[piece_str][end_measure+1] # ends on downbeat of next measure
            
            tup = (start_time, end_time, start_measure, end_measure, start_str, end_str)
            d[query_str].append(tup)
            
    return d

def save_query_info_to_file(d: dict[str, list[tuple[float, float, int, int, int, int]]], outfile: str = query_gt_file):
    """
    Write ground truth timestamps of queries contained in the input dictionary onto an output file
    (it is emptied before writing if it already exists)

    Params:
        d (dict[str, list[tuple[float, float, int, int, int, int]]]): dictionary indexed by query ids, containing a list of tuples with start and end time, start and end measure, 
                                                                      start and end line of each matched segment
        outfile (str): path to the output file
    """
    with open(outfile, 'w') as f:
        for query in sorted(d):
            print(f'Saving information about {query}')
            for (tstart, tend, mstart, mend, lstart, lend) in d[query]:
                f.write('{},{:.2f},{:.2f},{},{},{},{}\n'.format(query, tstart, tend, mstart, mend, lstart, lend))


