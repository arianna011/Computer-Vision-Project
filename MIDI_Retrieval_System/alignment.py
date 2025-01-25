import numpy as np
from numpy.matlib import repmat
import librosa as lb
import matplotlib.pyplot as plt
import os
from Cython_DTW import optimized_dtw as dtw

    
"""
Functions to compute the alignment between a query and a refernece bootleg score.
"""

def cost_metric(X,Y):
    """
    Compute the cost of aligning X and Y via negative dot product
    """
    return -1 * np.dot(X,Y)

def normalized_cost_metric(Q, R, num_query_notes, num_ref_notes):
    """
    Compute the cost of aligning X and Y via negative dot product
    and normalize it by dividing for the maximum between:
    - the number of simultaneous noteheads in sheet music
    - the number of simultaneous note onsets in the MIDI
    """
    cost = -1 * np.matmul(Q.T, R)
    query_norm_factor = repmat(num_query_notes.reshape((-1,1)), 1, R.shape[1]) # repeat the column to match R's number of columns
    ref_norm_factor = repmat(num_ref_notes.reshape((1,-1)), Q.shape[1], 1) # repeat the row to match Q's number of columns
    norm_factor = np.maximum(query_norm_factor, ref_norm_factor) + 1e-8 # avoid divide by 0
    norm_cost = cost / norm_factor
    return norm_cost

def align_bootleg_scores(query: np.ndarray, ref: np.ndarray, num_ref_notes: int, 
                         steps: list[int] = [1,1,1,2,2,1], weights: list[float] = [1,1,2], 
                         optimized: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the alignment between a query and a reference bootleg score via Dynamic Time Warping (DTW)

    Params:
        query (np.ndarray): the query bootleg score
        ref (np.ndarray): the reference bootleg score
        num_ref_notes (int): number of simultaneous note onsets in the MIDI (reference) bootleg
        steps (list[int]): list of step sizes allowed during DTW alignment, interpreted as pairs of (row_step, col_step)
        weights (list[float]): weights applied to each type of step
        optimized (bool): whether to use the optimized cython version of DTW

    Returns:
        (np.ndarray): the accumulated cost matrix computed by DTW, of dim = (n_query_frames x n_ref_frames);
                      each entry D[i, j] represents the minimum cost to align the first i frames of query with the first j frames of ref
        (np.ndarray): the optimal warping path, of dim = (n_steps x 2), represented as a sequence of (query_frame_index, ref_frame_index) pairs;
                      each pair shows which frame in the query aligns with which frame in the ref
    """
    if optimized: # Cython implementation
        # set params
        assert len(steps) % 2 == 0, "The length of steps must be even."
        dn = np.array(steps[::2], dtype=np.uint32) # row step sizes
        dm = np.array(steps[1::2], dtype=np.uint32) # column step sizes
        dw = weights
        subsequence = True
        parameter = {'dn': dn, 'dm': dm, 'dw': dw, 'SubSequence': subsequence}

        # Compute cost matrix
        num_query_notes = np.sum(query, axis=0)
        cost = normalized_cost_metric(query, ref, num_query_notes, num_ref_notes) # pairwise alignment cost for all frames

        # DTW
        [D, s] = dtw.DTW_cost_to_accum_cost_and_steps(cost, parameter)
        [wp, end_col, end_cost] = dtw.DTW_get_path(D, s, parameter) # optimal warping path (wp), column index of the endpoint (end_col), total cost (end_cost)

        # Reformat the output
        wp = wp.T[::-1] # transpose and reverse to present the path in the correct order
    
    else: # librosa implementation
        steps = np.array(steps).reshape((-1,2))
        D, wp = lb.sequence.dtw(query, ref, step_sizes_sigma = steps, weights_mul = weights, subseq = True, metric = cost_metric)
    
    return D, wp     


def plot_alignment(D: np.ndarray, wp: np.ndarray, seg_info: tuple[tuple[float,float], list[tuple[float,float]], list[tuple[float, float]]] = None, fig_sz: tuple[int, int] = (10,10)):
    """
    Plot the accumulated cost metric of an alignment with overlayed warping path and segment information (if provided)

    Params:
        D (np.ndarray): the accumulated cost matrix computed by DTW, of dim = (n_query_frames x n_ref_frames);
                        each entry D[i, j] represents the minimum cost to align the first i frames of query with the first j frames of ref
        wp (np.ndarray): the optimal warping path, of dim = (n_steps x 2), represented as a sequence of (query_frame_index, ref_frame_index) pairs;
                         each pair shows which frame in the query aligns with which frame in the ref
        seg_info (tuple[tuple[float,float], list[tuple[float,float]], list[tuple[float, float]]]): additional segment information for the reference and query sequences.
                                                                                                   It contains:
                                                                                                    - match_seg_time: (start_time, end_time) for the matched segment in the query.
                                                                                                    - ref_seg_times: list of (start_time, end_time) pairs for reference segments.
                                                                                                    - ref_seg_cols: list of (start_col, end_col) column indices for the corresponding reference segments
        fig_sz (tuple[int,int]): the figure size
    """
    plt.figure(figsize = fig_sz)
    plt.imshow(D, origin = 'lower', cmap = 'jet')
    plt.plot(wp[:,1], wp[:,0], color='y')
    plt.xlabel('Ref')
    plt.ylabel('Query')
    if seg_info is not None:
        match_seg_time, ref_seg_times, ref_seg_cols = seg_info
        for i,ref_seg_col in enumerate(ref_seg_cols):
            plt.axvline(ref_seg_col[0], color = 'm') # mark the boundaries of each segment with vertical lines
            plt.axvline(ref_seg_col[1], color = 'm')
            plt.title('Hyp ({:.1f} s, {:.1f} s), Ref ({:.1f} s, {:.1f} s)'.format(match_seg_time[0], match_seg_time[1], ref_seg_times[i][0], ref_seg_times[i][1]))
        else:
            plt.title('Subsequence DTW Alignment')   
    plt.show() 

def get_predicted_timestamps(wp: np.ndarray, times: list[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Extract the predicted start and end timestamps (in seconds and MIDI ticks) in the MIDI reference sequence 
    that correspond to the warping path's start and end points.

    Params:
        wp (np.ndarray): the optimal warping path, of dim = (n_steps x 2), represented as a sequence of (query_frame_index, ref_frame_index) pairs;
                             each pair shows which frame in the query aligns with which frame in the ref
        times (list[tuple[float, float]]): list of (tsec, ttick) tuples indicating the time in ticks and seconds for each event in a MIDI file

    Returns:
        (tuple[float, float]): predicted start and end timestamps in seconds
        (tuple[float, float]): predicted start and end timestamps in MIDI ticks
    """
    start_frm_midi = wp[-1,1] # last frame index in the reference sequence corresponding to the query
    end_frm_midi = wp[0,1] # first frame index in the reference sequence corresponding to the query
    start_time_midi = times[start_frm_midi][0] # in sec
    end_time_midi = times[end_frm_midi][0]
    start_tick_midi = times[start_frm_midi][1] # in ticks
    end_tick_midi = times[end_frm_midi][1]
    return (start_time_midi, end_time_midi), (start_tick_midi, end_tick_midi)

def get_ground_truth_timestamps(img_file: str, col2times: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], list[tuple[int,int]]]:
    """
    Extract ground truth start and end timestamps, as well as their corresponding column indices, for a query image 
    from a pre-defined ground truth file and a mapping of bootleg score columns to timestamps

    Params:
        img_file (str): path to the query image file
        col2times (list[tuple[float, float]]): list of (tsec, ttick) tuples indicating the time in ticks and seconds for each event in a MIDI file

    Returns:
        (list[tuple[float,float]]): ground truth start and end times for the query
        (list[tuple[int,int]]): corresponding start and end columns in the bootleg score matrix
    """
    # get ground truth start, end times
    query = os.path.splitext(os.path.basename(img_file))[0] # e.g. '/path/to/dir/p1_q10.jpg' -> 'p1_q10'
    query_gt_file = 'data/query_info/query.gt' # each line is expected to be in the format: query_id, start_time, end_time, ...
    ref_matches_time = []
    with open(query_gt_file, 'r') as f:
        for line in f:
            parts = line.rstrip().split(',')
            if parts[0] == query:
                t_start = float(parts[1])
                t_end = float(parts[2])
                ref_matches_time.append((t_start, t_end))

    # get start, end columns in bootleg score
    bscore_cols = np.arange(len(col2times))
    times = [tup[0] for tup in col2times]
    ref_matches_col = []
    for (t_start, t_end) in ref_matches_time:
        col_start, col_end = np.interp([t_start, t_end], times, bscore_cols)
        ref_matches_col.append((col_start, col_end))

    return ref_matches_time, ref_matches_col

def visualize_aligned_bootleg_scores(s1: np.ndarray, s2: np.ndarray, wp: np.ndarray, lines: list[int]):
    """
    Align and visualize two bootleg score sequences based on their warping path. 
    It stacks the two aligned sequences for combined visualization

    Params:
        s1 (np.ndarray): the first bootleg score sequence
        s2 (np.ndarray): the second bootleg score sequence
        wp (np.ndarray): warping path
        lines (list[int]): staff lines positions

    """
    idxs1 = wp[::-1, 0]
    warped1 = s1[:,idxs1]
    idxs2 = wp[::-1, 1]
    warped2 = s2[:,idxs2]
    stacked = np.vstack((warped2, warped1))
    all_lines = []
    all_lines.extend(lines)
    all_lines.extend(np.array(lines) + s1.shape[0])
    visualize_long_bootleg_score(stacked, all_lines)


def visualize_long_bootleg_score(bs: np.ndarray, lines: list[int], chunk_sz: int = 150):
    """
    Visualize on screen a bootleg score which is long in the horizontal dimension
    by dividing it in chunks and showing one at a time

    Params:
        bs (np.ndarray): bootleg score to visualize
        lines (list[int]): staff lines positions
        chunk_sz (int): size of each chunk
    """
    chunks = bs.shape[1] // chunk_sz + 1
    for i in range(chunks):
        start_col = i * chunk_sz
        end_col = min((i + 1)*chunk_sz, bs.shape[1])
        visualize_bootleg_score(bs[:,start_col:end_col], lines)
        

def visualize_bootleg_score(bs: np.ndarray, lines: list[int], fig_sz: tuple[int, int] = (10,10)):
    """
    Visualize on screen a bootleg score with staff lines.

    Params:
        bs (np.ndarray): the bootleg score to visualize
        ines (list[int]): the positions of staff lines for both left and right hands
        fig_sz (tuple[int, int]): the size of the figure to show
    """
    plt.figure(figsize = fig_sz)
    plt.imshow(1 - bs, cmap = 'gray', origin = 'lower')
    for l in range(1, bs.shape[0], 2):
        plt.axhline(l, c = 'grey')
    for l in lines:
        plt.axhline(l, c = 'r')
    plt.show()