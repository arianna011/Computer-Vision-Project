from MIDI_Retrieval_System import BootlegScore, MIDIProcessing, QueryProcessing, MusicalObjectDetection
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import process_data
import evaluation as eval
import pdf2image
from main import find_image, find_pdf, main
import time
from multiprocessing import Pool
from multiprocessing import cpu_count

def test_bootleg_score(midi_file):
    # visualize bootleg score
    midi = MIDIProcessing(midi_file)
    result, _ =  midi.get_note_events(quant=10)
    both, times, numNotes, _, _ = midi.generate_bootleg_score(result, 2, 2)
    bs = BootlegScore(both[:,0:140])
    bs.visualize(staff_lines=MIDIProcessing.staff_lines_both)

def test_midi_processing():
    # process midi batch
    fileList = './cfg_files/midi.train.list' # list of all midi files to process
    outdir = 'experiments/train/db' # where to save bootleg scores
    MIDIProcessing.process_midi_batch(fileList, outdir)


def test_all_query_bootleg_generation(img_file, verbose=True):

    # PRE-PROCESSING
    proc = QueryProcessing(img_file)
    det = proc.assign_detector()

    proc.normalize_pre_processed_image()
    if verbose: QueryProcessing.show_grayscale_image(proc.norm_inv_img)

    # isolate staff lines
    det.isolate_staff_lines(MusicalObjectDetection.morph_filter_rect_len, 
                            MusicalObjectDetection.notebar_filter_len, 
                            MusicalObjectDetection.notebar_removal)
    hlines = det.isol_staff_lines
    if verbose: QueryProcessing.show_grayscale_image(hlines)

    # compute staff features
    staff_featmap, stave_lens, col_w = det.compute_staff_feature_map(MusicalObjectDetection.stave_feat_map_n_cols, 
                                                                     MusicalObjectDetection.stave_feat_map_lower_bound, 
                                                                     MusicalObjectDetection.stave_feat_map_upper_bound, 
                                                                     MusicalObjectDetection.stave_feat_map_step)
    if verbose:
        print("Feature Map Shape:", staff_featmap.shape)
        print("Stave Lengths:", stave_lens)
        print("Column Width:", col_w)
    
    # NOTEHEAD DETECTION
    keypoints, img_with_keypoints = det.detect_notehead_blobs(min_area=MusicalObjectDetection.note_detect_min_area, 
                                                              max_area = MusicalObjectDetection.note_detect_max_area)
    if verbose: proc.show_color_image(img_with_keypoints)

    note_template, n_crops = det.get_note_template(keypoints, MusicalObjectDetection.note_template_size)
    if verbose:
        print(f'Number of crops: {n_crops}')
        proc.show_grayscale_image(note_template, (3,3))

    _, img_bin_notes = det.adaptive_notehead_detect(note_template, MusicalObjectDetection.note_detect_tol_ratio, MusicalObjectDetection.chord_specs)
    if verbose: proc.show_img_with_bound_boxes(img_bin_notes, det.notes_bboxes)

    note_centers, h_mean, w_mean = det.get_notehead_info()
    if verbose:
        print("Average height: ", h_mean)
        print("Average width: ", w_mean)

    # INFER NOTES VALUES

    # local staff estimation
    est_staff_lines, staff_len = QueryProcessing.estimate_staff_line_locs(staff_featmap, note_centers, stave_lens, col_w, QueryProcessing.max_delta_row_initial, int(-2*QueryProcessing.target_line_sep))
    if verbose:
        QueryProcessing.visualize_pred_staff_lines(est_staff_lines, hlines)
        print(f'Estimated staff length: {staff_len}')

    # global staff midpoints clustering
    stave_mid_pts = QueryProcessing.estimate_staff_midpoints(est_staff_lines, QueryProcessing.min_num_staves, QueryProcessing.max_num_staves, QueryProcessing.min_stave_separation)
    stave_idxs, nh_row_offsets = QueryProcessing.assign_noteheads_to_staves(note_centers, stave_mid_pts)
    if verbose: QueryProcessing.visualize_clusters(proc.norm_inv_img, note_centers, stave_idxs)

    # refined staff lines estimation
    est_staff_line_locs, staff_len = QueryProcessing.estimate_staff_line_locs(staff_featmap, note_centers, stave_lens, col_w, 
                                                                              QueryProcessing.max_delta_row_refined, 
                                                                              (nh_row_offsets-2*QueryProcessing.target_line_sep).astype(int))
    if verbose:
        QueryProcessing.visualize_pred_staff_lines(est_staff_line_locs, hlines)
        print(f'Estimated staff length: {staff_len}')

    # note labeling estimation
    nh_vals = QueryProcessing.estimate_note_labels(est_staff_line_locs)
    if verbose: QueryProcessing.visualize_note_labels(proc.norm_inv_img, nh_vals, note_centers)

    # CLUSTER NOTES AND STAVES

    det.isolate_bar_lines(MusicalObjectDetection.morph_filter_bar_vert, MusicalObjectDetection.morph_filter_bar_hor, MusicalObjectDetection.max_barline_width)
    vlines = det.isol_bar_lines
    if verbose: proc.show_grayscale_image(vlines)

    # compute staff midpoints
    stave_map, evidence = QueryProcessing.determine_stave_grouping(stave_mid_pts, vlines)
    if verbose:
        print(f'Evidences median: ({np.median(evidence[2])}, {np.median(evidence[3])})')
        plt.plot(np.sum(vlines, axis=1))
        for m in stave_mid_pts:
            plt.axvline(m, color = 'r')
        plt.show()
    note_clusters, clusters_pairs = QueryProcessing.cluster_noteheads(stave_idxs, stave_map)
    if verbose: QueryProcessing.visualize_clusters(proc.norm_inv_img, note_centers, note_clusters)

    # BOOTLEG SCORE GENERATION

    note_data = [(int(np.round(note_centers[i][0])), int(np.round(note_centers[i][1])), nh_vals[i], note_clusters[i]) for i in range(len(note_centers))]

    bscore_query, events, event_idxs = QueryProcessing.generate_query_bootleg_score(note_data, clusters_pairs, min_col_diff=w_mean, 
                                                                                    repeat_notes=QueryProcessing.bootleg_repeat_notes, filler=QueryProcessing.bootleg_filler)
    
    QueryProcessing.visualize_long_bootleg_score(bscore_query, QueryProcessing.staff_lines_both)


def test_all_midi_retrieval():
    start = time.time()
    queries = [os.path.join('data/queries', q) for q in os.listdir('data/queries')]
    queries = queries[:200]
    c=0
    for q in queries:
        midi, interval = find_image(q)
        if not _correct_predict(midi.filename, q):
            #print(f'Failed query: {q}, Matched MIDI: {midi.filename}')
            c += 1 

    print(f'Total Errors: {c}')
    print(f'Total Queries: {len(queries)}')
    print(f'Accuracy: {1 - c/len(queries)}')
    end = time.time()
    print(f"Runtime: {end - start:.4f} secondi")
    print(queries)


def test_all_pdf_retrieval():
    start = time.time()
    queries = [os.path.join('data/queries', q) for q in os.listdir('data/queries')]
    for q in queries:
        pdf = main(q, "PDF")
    end = time.time()
    print(f"Runtime: {end - start:.4f} secondi")

def _correct_predict(midi_name, query_name):
    query_name = query_name.split('_')[0]
    piece_query = "".join([c for c in query_name if c.isdigit()])
    piece_midi = "".join([c for c in midi_name if c.isdigit()])
    #print(f'Query: {piece_query}, MIDI: {piece_midi}')
    return piece_query == piece_midi
   

if __name__ == "__main__":

    # random examples
    midi_file = './data/midi/p91.mid'
    img_file = 'data/queries/p138_q6.jpg'
    midi_db_dir = 'experiments/train/db'

    #test_all_query_bootleg_generation(img_file, verbose=False)

    #bs_score_midi = BootlegScore.build_from_midi(midi_file)
    #bs_score_midi.visualize_long(MIDIProcessing.staff_lines_both, chuncks_sz=500) # many images

    #bs_score_query = BootlegScore.build_from_img(img_file)
    #bs_score_query.visualize(QueryProcessing.staff_lines_both)

    ### save pdfs' bootleg scores as pickle files
    #process_data.process_all_pdfs()

    ### test pdf to image
    #images = pdf2image.convert_from_path('./data/pdfs/p1.pdf', dpi=300)
    #find_image(images[0], 'MIDI')

    ### test find pdf
    find_pdf(img_file)

    #start = time.time()
    #midi, interval = find_image(img_file)
    #end = time.time()

    #print(f"Runtime: {end - start:.4f} secondi")
    #print(f"MIDI: {midi.filename}, Interval: {interval}")
    

    #test_all_midi_retrieval()
    #test_all_pdf_retrieval()
