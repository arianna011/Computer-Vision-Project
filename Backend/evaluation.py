import process_data as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

"""
Functions to evaluate the MIDI Retrieval System
and to investigate errors
"""

def read_ground_truth_labels(gt_file: str = pd.query_gt_file) -> dict[str, list[tuple[float, float]]]:
    """
    Read the ground truth timestamps (start and end) of the matched segments in the MIDI piece 
    corresponding to each query from the input file

    Params:
        gt_file (str): path to the file containing ground truth query information

    Returns:
        (dict[str, list[tuple[float, float]]]): dictionary indexed by query ids containing lists of start and end time
                                                of matched segments in the associated MIDI file
    """
    d = {}
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.rstrip().split(',') # e.g. 'p1_q1,1.55,32.59'
            query_str = parts[0]
            t_start = float(parts[1])
            t_end = float(parts[2])
            if query_str in d:
                d[query_str].append((t_start, t_end))
            else:
                d[query_str] = [(t_start, t_end)]
    return d


def read_hypothesis_files(hyp_dir: str) -> list[tuple[str, float, float]]:
    """
    Read the predicted timestamps (start and end) of the matched segments in the MIDI piece 
    corresponding to each query from the files contained in the input directory

    Params:
        hyp_dir (str): path to the directory containing files with predicted timestamps
    
    Returns:
        (list[tuple[str, float, float]]): a list of tuples including query ids and predicted start and end time of matched segment
    """
    l = []
    for hypfile in sorted(glob.glob("{}/*.hyp".format(hyp_dir))):
        with open(hypfile, 'r') as f:
            line = next(f)
            parts = line.rstrip().split(',')
            query = parts[0]  # e.g. p1_q1
            t_start = float(parts[1])
            t_end = float(parts[2])
            l.append((query, t_start, t_end))
    return l


def compute_precision_recall(hyp_dir: str = pd.out_dir, 
                             gt_file: str = pd.query_gt_file) -> list[tuple[str, float, float, float, int]]:
    """
    Compute the F-score, precision and recall for the predictions about matched segments timestamps in the input directory
    based on the ground truth timestamps in the input file

    Params:
        hyp_dir (str): path to the directory containing files with predicted timestamps
        gt_file (str): path to the file containing ground truth query information

    Returns:
        (float): F-score (harmonic mean of precision and recall)
        (float): precision (total amount of overlap between predicted and ground truth timestamps divided by the duration of predicted time intervals)
        (float): recall (total amount of overlap between predicted and ground truth timestamps divided by the duration of ground truth time intervals)
        (list[tuple[str, float, float, float, int]]): list of tuples including query ids, maximum overlap with a reference segment, 
                                                      start and end timestamps of such segment and its index in the list of matched segments to the query
    """
    d = read_ground_truth_labels(gt_file)
    hyps = read_hypothesis_files(hyp_dir)
    hypinfo = [] 
    overlap_total, hyp_total, ref_total = (0,0,0)
    for (query_id, hyp_start, hyp_end) in hyps:
        ref_segments = d[query_id]
        idx_max = 0
        overlap_max = 0
        for i, ref_seg in enumerate(ref_segments): # find ref segment with most overlap
            overlap = compute_overlap((hyp_start, hyp_end), ref_seg)
            if overlap > overlap_max:
                idx_max = i
                overlap_max = overlap
        hyplen = hyp_end - hyp_start
        reflen = ref_segments[idx_max][1] - ref_segments[idx_max][0]        
        overlap_total += overlap_max
        hyp_total += hyplen
        ref_total += reflen
        hypinfo.append((query_id, overlap_max, ref_segments[idx_max][0], ref_segments[idx_max][1], idx_max)) # keep for error analysis
    P = overlap_total / hyp_total
    R = overlap_total / ref_total
    F = 2 * P * R / (P + R)
    return F, P, R, hypinfo


def compute_overlap(seg_1: tuple[float, float], seg_2: tuple[float, float]) -> float:
    """
    Compute the overlap between two time intervals
    """
    overlap_lb = max(seg_1[0], seg_2[0])
    overlap_ub = min(seg_1[1], seg_2[1])
    overlap = np.clip(overlap_ub - overlap_lb, 0, None)
    return overlap  


def print_debugging_info(score_info: dict[int, dict[int, tuple[int,int]]], midi_info: dict[int, dict[int, float]], 
                       query_info: dict[str, list[tuple[float, float, int, int, int, int]]], hyp_info: list[tuple[str, float, float, float, int]], 
                       hyp_dir: str = pd.out_dir, gt_file: str = pd.query_gt_file):
    """
    Test the MIDI Retrieval System and print debugging information such as overlap between predicted and groundtruth matched segments between queries and MIDIs
    in terms of time, measures and lines in the scores 

    Params:
        score_info (dict[int, dict[int, tuple[int,int]]]): dictonary indexed by piece numbers containing
                                                           dictionaries with line numbers as keys and tuples of start measure and end measure as values
        midi_info (dict[int, dict[int, float]]): dictonary indexed by piece numbers containing
                                                 dictionaries with measure numbers as keys and time duration as values
                                                 (an additional entry identified by the last measure number + 1
                                                  contains the total duration of the MIDI piece))
        query_info (dict[str, list[tuple[float, float, int, int, int, int]]]): dictionary indexed by query ids, containing a list of tuples with start and end time, start and end measure, 
                                                                               start and end line of each matched segment
        hyp_info (list[tuple[str, float, float, float, int]]): list of tuples including query ids, maximum overlap with a reference segment, 
                                                               start and end timestamps of such segment and its index in the list of matched segments to the query
        hyp_dir (str): path to the directory containing files with predicted timestamps
        gt_file (str): path to the file containing ground truth query information
    """
    d = read_ground_truth_labels(gt_file)
    hyps = read_hypothesis_files(hyp_dir)

    for i, (query, hyp_tstart, hyp_tend) in enumerate(hyps):
        
        # hyp and ref info (sec)
        piece = query.split('_')[0]
        _, overlap, ref_tstart, ref_tend, best_idx = hyp_info[i]
        
        # hyp and ref info (measures)
        interp_m = list(midi_info[piece].keys())
        interp_t = [midi_info[piece][m] for m in interp_m]
        hyp_mstart, hyp_mend, ref_mstart, ref_mend = np.interp([hyp_tstart, hyp_tend, ref_tstart, ref_tend], interp_t, interp_m)
        m_overlap = compute_overlap((hyp_mstart, hyp_mend),(ref_mstart, ref_mend))
        
        # hyp and ref info (line # + measure offset)
        hyp_lstart, hyp_lstartoff = get_line_number_measure_offset(hyp_mstart, score_info[piece])
        hyp_lend, hyp_lendoff = get_line_number_measure_offset(hyp_mend, score_info[piece])
        ref_lstart = query_info[query][best_idx][4]
        ref_lend = query_info[query][best_idx][5]
        
        # compare in sec
        print("{}: hyp ({:.1f} s,{:.1f} s), ref ({:.1f} s,{:.1f} s), overlap {:.1f} of {:.1f} s".format(query, hyp_tstart, hyp_tend, ref_tstart, ref_tend, overlap, ref_tend - ref_tstart))
        
        # compare in measure numbers
        print("\thyp ({:.1f} m, {:.1f} m), ref ({:.1f} m, {:.1f} m), overlap {:.1f} m".format(hyp_mstart, hyp_mend, ref_mstart, ref_mend, m_overlap))
        
        # compare in line + measure offset
        print("\thyp (ln {} m{:.1f}, ln {} m{:.1f}), ref (ln {}, ln {})".format(hyp_lstart, hyp_lstartoff, hyp_lend, hyp_lendoff, ref_lstart, ref_lend))
    return


def get_line_number_measure_offset(measure_num: int, piece_info: dict[int, tuple[int,int]]) -> tuple[int, int]:
    """
    Retrieve the offset of a measure with respect to a line in the score of piece

    Params:
        measure_num (int): measure id
        piece_info (dict[int, tuple[int,int]]): dictionary with line numbers as keys and tuples of start measure and end measure as values

    Returns:
        (int): the line of the piece score where the measure is
        (int): the offset (in terms of number of measures) in the line where the measure is
    """
    line = -1
    m_offset = -1
    for key in piece_info:
        lb, ub = piece_info[key] # line start, end measure 
        if measure_num >= lb and measure_num < ub + 1:
            line = key
            m_offset = measure_num - lb + 1
            break
    return line, m_offset    


def show_runtime_stats(hyp_dir: str = pd.out_dir):
    """
    Read the runtime information from the files in the input directory
    and print them on screen

    Params:
        hyp_dir (str): path to the directory containing files with predicted timestamps
    """
    durs = []
    cnt = 0
    for hypfile in glob.glob('{}/*.hyp'.format(hyp_dir)):
        cnt += 1
        with open(hypfile, 'r') as f:
            line = next(f)
            parts = line.split(',')
            dur = float(parts[3])
            durs.append(dur)
    durs = np.array(durs)
    avg_dur = np.mean(durs)
    min_dur = np.min(durs)
    max_dur = np.max(durs)
    std_dur = np.std(durs)
    print('{} files'.format(cnt))
    print('Avg Duration: {:.2f} sec'.format(avg_dur))
    print('Std Duration: {:.2f} sec'.format(std_dur))
    print('Min Duration: {:.2f} sec'.format(min_dur))
    print('Max Duration: {:.2f} sec'.format(max_dur))
    plt.hist(durs, bins=np.arange(min_dur,max_dur))
    plt.xlabel('Runtime (sec)')
    plt.ylabel('Count')
    plt.show()


if __name__ == "__main__":
    
    # eval training set
    # pd.process_all_midis()
    # outs = pd.process_all_queries() # takes a while

    score_info = pd.import_score_info()
    midi_info = pd.import_midi_info()
    query_info = pd.get_query_ground_truth(score_info, midi_info)
    pd.save_query_info_to_file(query_info)

    # F, P, R, hyp_info = compute_precision_recall()
    # print(f'TRAINING F-score: {F}, Precision: {P}, Recall: {R}')
    # #print_debugging_info(score_info, midi_info, query_info, hyp_info)
    # show_runtime_stats()

    # eval test set
    pd.process_all_midis('cfg_files/midi.test.list', 'experiments/test/db')
    outs = pd.process_all_queries('cfg_files/query.test.list', 'experiments/test/db', 'experiments/test/hyp') # takes a lot

    F, P, R, hyp_info = compute_precision_recall('experiments/test/hyp', 'data/query_info/query.gt')
    print(f'TEST F-score: {F}, Precision: {P}, Recall: {R}')
    #print_debugging_info(score_info, midi_info, query_info, hyp_info, 'experiments/test/hyp', 'data/query_info/query.gt')
    show_runtime_stats('experiments/test/hyp')