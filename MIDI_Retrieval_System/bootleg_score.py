import numpy as np
import matplotlib.pyplot as plt
from MIDI_Retrieval_System.processing_image import QueryProcessing
from MIDI_Retrieval_System.processing_midi import MIDIProcessing

class BootlegScore:
    """
    Feature representation to encode the position of noteheads in relation to staff lines in sheet music.
    """
    
    def __init__(self, X):
        """
        Initialize the BootlegScore with a NumPy array representing the bootleg score.
        
        Parameters:
        - X (numpy.ndarray): 2D array representing the bootleg score image.
        """
        self.X = X


    def visualize(self, staff_lines, sz=(10,10)):
        """
        Show the bootleg score as an image containing 
        black rectangulars noteheads placed on standard horizontal blue lines 
        and red staff lines (provided as input)
        """
        plt.figure(figsize = sz)
        plt.imshow(1 - self.X, cmap = 'gray', origin = 'lower') # invert the colors of the bootleg score 

        for l in range(1, self.X.shape[0], 2):  # Draw blue lines at every second row
            plt.axhline(l, c='b')

        for l in staff_lines:  # Draw red lines at specified staff line positions
            plt.axhline(l, c='r')

        plt.show()

    def visualize_long(self, staff_lines, chuncks_sz):
        """
        Visualize on screen a bootleg score which is long in the horizontal dimension
        by dividing it in chunks and showing one at a time
        """
        QueryProcessing.visualize_long_bootleg_score(self.X, staff_lines, chuncks_sz)

    @staticmethod
    def build_from_midi(midi_file):
        """
        Generate the bootleg score corresponding to the input MIDI file

        Params:
            midi_file (str): path of the input MIDI

        Returns:
            (BootlegScore): a BootlegScore object generated from the input MIDI
        """
        midi = MIDIProcessing(midi_file)
        events, _ =  midi.get_note_events(quant=midi.time_quant_factor)
        bs, _, _, _, _ = midi.generate_bootleg_score(events, 2, 2)
        return BootlegScore(bs)


    @staticmethod
    def build_from_img(img_file):
        """
        Generate the bootleg score corresponding to the sheet music in the input image file

        Params:
            img_file (str): path of the input image

        Returns:
            (BootlegScore): a BootlegScore object generated from the input image
        """

        # IMAGE PRE-PROCESSING
        proc = QueryProcessing(img_file)
        det = proc.assign_detector()
        proc.normalize_pre_processed_image()

        # isolate staff lines
        det.isolate_staff_lines(det.morph_filter_rect_len, 
                                det.notebar_filter_len, 
                                det.notebar_removal)
    
        # compute staff features
        staff_featmap, stave_lens, col_w = det.compute_staff_feature_map(det.stave_feat_map_n_cols, 
                                                                         det.stave_feat_map_lower_bound, 
                                                                         det.stave_feat_map_upper_bound, 
                                                                         det.stave_feat_map_step)
        
        # NOTEHEAD DETECTION
        keypoints, _ = det.detect_notehead_blobs(min_area=det.note_detect_min_area, 
                                                                  max_area = det.note_detect_max_area)
        note_template, _ = det.get_note_template(keypoints, det.note_template_size)
        _, img_bin_notes = det.adaptive_notehead_detect(note_template, det.note_detect_tol_ratio, det.chord_specs)
        note_centers, h_mean, w_mean = det.get_notehead_info()

        # INFER NOTES VALUES

        # local staff estimation
        est_staff_lines, staff_len = QueryProcessing.estimate_staff_line_locs(staff_featmap, note_centers, stave_lens, col_w, 
                                                                              QueryProcessing.max_delta_row_initial, int(-2*QueryProcessing.target_line_sep))

        # global staff midpoints clustering
        stave_mid_pts = QueryProcessing.estimate_staff_midpoints(est_staff_lines, QueryProcessing.min_num_staves, QueryProcessing.max_num_staves, QueryProcessing.min_stave_separation)
        stave_idxs, nh_row_offsets = QueryProcessing.assign_noteheads_to_staves(note_centers, stave_mid_pts)

        # refined staff lines estimation
        est_staff_line_locs, staff_len = QueryProcessing.estimate_staff_line_locs(staff_featmap, note_centers, stave_lens, col_w, 
                                                                                QueryProcessing.max_delta_row_refined, 
                                                                                (nh_row_offsets-2*QueryProcessing.target_line_sep).astype(int))

        # note labeling estimation
        nh_vals = QueryProcessing.estimate_note_labels(est_staff_line_locs)

        # CLUSTER NOTES AND STAVES

        det.isolate_bar_lines(det.morph_filter_bar_vert, det.morph_filter_bar_hor, det.max_barline_width)
        vlines = det.isol_bar_lines

        # compute staff midpoints
        stave_map, evidence = QueryProcessing.determine_stave_grouping(stave_mid_pts, vlines)
        note_clusters, clusters_pairs = QueryProcessing.cluster_noteheads(stave_idxs, stave_map)

        # BOOTLEG SCORE GENERATION
        note_data = [(int(np.round(note_centers[i][0])), int(np.round(note_centers[i][1])), nh_vals[i], note_clusters[i]) for i in range(len(note_centers))]
        bscore_query, _, _ = QueryProcessing.generate_query_bootleg_score(note_data, clusters_pairs, min_col_diff=w_mean, 
                                                                                        repeat_notes=QueryProcessing.bootleg_repeat_notes, filler=QueryProcessing.bootleg_filler)
        
        return BootlegScore(bscore_query)
    


    


