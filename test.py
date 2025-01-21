from MIDI_Retrieval_System import BootlegScore, MIDIProcessing, QueryProcessing, MusicalObjectDetection
import matplotlib.pyplot as plt

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

def test_query_pre_processing(img_file):
    # visualize picture pre-processing step-by-step
    q_proc = QueryProcessing(img_file)
    new_img = q_proc.remove_background_lightning()
    plt.imshow(q_proc.img)
    plt.show()
    plt.imshow(new_img)
    plt.show()
    line_sep, scores = q_proc.estimate_line_sep(QueryProcessing.est_line_sep_n_cols, QueryProcessing.est_line_sep_lower_bound, QueryProcessing.est_line_sep_upper_bound, QueryProcessing.est_line_sep_step)
    print(f'Estimated line sep: {line_sep}')
    h, w = q_proc.calculate_resized_dimensions(line_sep, QueryProcessing.target_line_sep)
    print(f'Target dims: {h} x {w}')
    resized_img = new_img.resize((w,h))
    plt.imshow(resized_img)
    plt.show()
    print(f'Picture dims: {resized_img.height} x {resized_img.width}')

def test_query_pre_processing_one_step(img_file):
    # test the function that does all query pre-processing by itself
    q_proc = QueryProcessing(img_file)
    q_proc.pre_process_image()
    img = q_proc.pre_processed_image
    plt.imshow(img)
    plt.show()
    print(f'Picture dims: {img.height} x {img.width}')


###########  TEST MUSICAL OBJECT DETECTION ####################

def test_staff_lines_detection(img_file):
    proc = QueryProcessing(img_file)
    det = proc.assign_detector()
    res = det.isolate_staff_lines(MusicalObjectDetection.morph_filter_rect_len, 
                                  MusicalObjectDetection.notebar_filter_len, 
                                  MusicalObjectDetection.notebar_removal)
    proc.show_grayscale_image(res)

    # Compute the staff feature map
    featmap, stave_lens, col_w = MusicalObjectDetection.compute_staff_feature_map(res, 
                                                               MusicalObjectDetection.stave_feat_map_n_cols, 
                                                               MusicalObjectDetection.stave_feat_map_lower_bound, 
                                                               MusicalObjectDetection.stave_feat_map_upper_bound, 
                                                               MusicalObjectDetection.stave_feat_map_step)
    # Display the results
    print("Feature Map Shape:", featmap.shape)
    print("Stave Lengths:", stave_lens)
    print("Column Width:", col_w)
    proc.show_grayscale_image(featmap[0])


def test_notehead_detection(img_file):

    proc = QueryProcessing(img_file)
    det = proc.assign_detector()

    # test erosion and dilation
    res = MusicalObjectDetection.morph_filter_circle(det.img, MusicalObjectDetection.morph_filter_circ_dilate, MusicalObjectDetection.morph_filter_circ_erode)
    proc.show_grayscale_image(res, max_val=255, inverted=False)

    # test blob detector
    keypoints, img_with_keypoints = det.detect_notehead_blobs(min_area=MusicalObjectDetection.note_detect_min_area, 
                                                              max_area = MusicalObjectDetection.note_detect_max_area)
    proc.show_color_image(img_with_keypoints)

    # test notehead template computing
    note_template, n_crops = det.get_note_template(keypoints, MusicalObjectDetection.note_template_size)
    print(f'Number of crops: {n_crops}')
    proc.show_grayscale_image(note_template, (3,3))

    # test adaptive notehead detection
    notes, img_bin_notes = det.adaptive_notehead_detect(note_template, MusicalObjectDetection.note_detect_tol_ratio, MusicalObjectDetection.chord_specs)
    proc.show_img_with_bound_boxes(img_bin_notes, notes)

    coords, h_mean, w_mean = MusicalObjectDetection.get_notehead_info(notes)
    print("Center coordinates: ", coords)
    print("Average height: ", h_mean)
    print("Average width: ", w_mean)


def test_barline_detection(img_file):
    proc = QueryProcessing(img_file)
    det = proc.assign_detector()
    vlines = det.isolate_bar_lines(MusicalObjectDetection.morph_filter_bar_vert, MusicalObjectDetection.morph_filter_bar_hor, MusicalObjectDetection.max_barline_width)
    proc.show_grayscale_image(vlines)


##################### TEST QUERY BOOTLEG PROJECTION ##########################

def test_local_staff_estimation(img_file):
    proc = QueryProcessing(img_file)
    det = proc.assign_detector()
    hlines = det.isolate_staff_lines(MusicalObjectDetection.morph_filter_rect_len, 
                                  MusicalObjectDetection.notebar_filter_len, 
                                  MusicalObjectDetection.notebar_removal)
    # Detect noteheads
    hlines = MusicalObjectDetection.morph_filter_circle(det.img, MusicalObjectDetection.morph_filter_circ_dilate, MusicalObjectDetection.morph_filter_circ_erode)
    keypoints, _ = det.detect_notehead_blobs(min_area=MusicalObjectDetection.note_detect_min_area, max_area = MusicalObjectDetection.note_detect_max_area)

    # test notehead template computing
    note_template, _ = det.get_note_template(keypoints, MusicalObjectDetection.note_template_size)
    notes, _ = det.adaptive_notehead_detect(note_template, MusicalObjectDetection.note_detect_tol_ratio, MusicalObjectDetection.chord_specs)

    # Compute the staff feature map
    featmap, stave_lens, col_w = MusicalObjectDetection.compute_staff_feature_map(hlines, 
                                                               MusicalObjectDetection.stave_feat_map_n_cols, 
                                                               MusicalObjectDetection.stave_feat_map_lower_bound, 
                                                               MusicalObjectDetection.stave_feat_map_upper_bound, 
                                                               MusicalObjectDetection.stave_feat_map_step)
    


    est_staff_lines, staff_len = QueryProcessing.estimate_staff_line_locs(featmap, notes, stave_lens, col_w, QueryProcessing.max_delta_row_initial, int(-2*QueryProcessing.target_line_sep))
    QueryProcessing.visualize_pred_staff_lines(est_staff_lines, hlines)
    print(f'Estimated staff length: {staff_len}')
    return hlines, featmap, notes, stave_lens, col_w, est_staff_lines


def test_global_staff_estimation(img_file):
    proc = QueryProcessing(img_file)
    det = proc.assign_detector()
    norm = proc.get_normalized_pre_processed_image()

    hlines, featmap, notes, stave_lens, col_w, est_staff_lines = test_local_staff_estimation(img_file)

    # test staff midpoints clustering
    stave_mid_pts = QueryProcessing.estimate_staff_midpoints(est_staff_lines, QueryProcessing.min_num_staves, QueryProcessing.max_num_staves, QueryProcessing.min_stave_separation)
    QueryProcessing.visualize_staff_midpoint_clustering(est_staff_lines, stave_mid_pts)
    stave_idxs, nh_row_offsets = QueryProcessing.assign_noteheads_to_staves(notes, stave_mid_pts)
    QueryProcessing.visualize_clusters(norm, notes, stave_idxs)

    # test refined staff lines estimation
    est_staff_line_locs, staff_len = QueryProcessing.estimate_staff_line_locs(featmap, notes, stave_lens, col_w, 
                                                                                 QueryProcessing.max_delta_row_refined, 
                                                                                 (nh_row_offsets-2*QueryProcessing.target_line_sep).astype(int))
    QueryProcessing.visualize_pred_staff_lines(est_staff_line_locs, hlines)
    print(f'Estimated staff length: {staff_len}')

    # test note labeling estimation
    nh_vals = QueryProcessing.estimate_note_labels(est_staff_line_locs)
    QueryProcessing.visualize_note_labels(norm, nh_vals, notes)

if __name__ == "__main__":

    # random examples
    midi_file = './data/midi/p91.mid'
    img_file = 'data/queries/p1_q1.jpg'
    midi_db_dir = 'experiments/train/db'

    #test_staff_lines_detection(img_file)
    #test_notehead_detection(img_file)
    #test_barline_detection(img_file)
    #test_local_staff_estimation(img_file)
    test_global_staff_estimation(img_file)

    
    
    
