import numpy as np
import matplotlib.pyplot as plt
import pickle
from MIDI_Retrieval_System.processing_image import QueryProcessing
from MIDI_Retrieval_System.processing_midi import MIDIProcessing
from MIDI_Retrieval_System import alignment as align

class BootlegScore:
    """
    Feature representation to encode the position of noteheads in relation to staff lines in sheet music.
    """

    staff_lines = [13,15,17,19,21,35,37,39,41,43] # locations of staff lines for both right and left hand
    # Alignment parameters
    dtw_steps = [1,1,2,1,1,2]
    dtw_weights = [1,1,2]

    def __init__(self, X):
        """
        Initialize the BootlegScore with a NumPy array representing the bootleg score.
        
        Parameters:
        - X (numpy.ndarray): 2D array representing the bootleg score image
        """
        self.X = X
        self.type = None # can be "MIDI" or "Image", indicates the source of the bootleg

        # parameters for "MIDI" bootleg score
        self.times = None
        self.num_notes = None
        self.note_events = None

        # parameters for "Image" bootleg score
        self.img_file = None # path of the original image

        # parameters for alignment between this and another bootleg score
        self.aligned_to = None # last bootleg score this bootleg has been aligned to
        self.cost_matr = None       # returned by DTW
        self.warping_path = None    # returned by DTW


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
        bs, times, num_notes, _, _ = midi.generate_bootleg_score(events, 2, 2)
        obj = BootlegScore(bs)
        obj.type = "MIDI"
        obj.times = times
        obj.num_notes = num_notes
        return obj

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
        if len(keypoints) == 0:
            print("No noteheads detected in the image")
            return None
        note_template, _ = det.get_note_template(keypoints, det.note_template_size)
        notes, _ = det.adaptive_notehead_detect(note_template, det.note_detect_tol_ratio, det.chord_specs)

        if len(notes) < QueryProcessing.max_num_staves: # if few or no notes detected, stop early (avoids later errors during kmeans clustering)
            print("No notes detected in the image")
            return None

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
        
        obj = BootlegScore(bscore_query)
        obj.type = "Image"
        obj.img_file = img_file
        return obj
    
    @staticmethod
    def load_midi_bootleg(pkl_file: str):
        """
        Load the bootleg score corresponding to a MIDI file stored in the input pickle file

        Params:
            pkl_file (str): path of the pickle file

        Returns:
            (BootlegScore): a BootlegScore object loaded from the input file
        """
        with open(pkl_file, 'rb') as f:
            d = pickle.load(f)
        bscore = d['bscore']
        miditimes = d['times']
        num_notes = np.array(d['num_notes'])
        obj = BootlegScore(bscore)
        obj.type = "MIDI"
        obj.times = miditimes
        obj.num_notes = num_notes
        return obj

    @staticmethod
    def load_img_bootleg(pkl_file: str):
        """
        Load the bootleg score corresponding to an image stored in the input pickle file

        Params:
            pkl_file (str): path of the pickle file

        Returns:
            (BootlegScore): a BootlegScore object loaded from the input file
        """
        with open(pkl_file, 'rb') as f:
            d = pickle.load(f)
        bscore = d['bscore']
        imgfile = d['image_file']
        obj = BootlegScore(bscore)
        obj.type = "Image"
        obj.img_file = imgfile
        return obj

    @staticmethod
    def load_pdf_bootleg(pkl_file: str):
        """
        Load the bootleg score corresponding to a PDF file stored in the input pickle file

        Params:
            pkl_file (str): path of the pickle file

        Returns:
            (BootlegScore): a BootlegScore object loaded from the input file
        """
        with open(pkl_file, 'rb') as f:
            d = pickle.load(f)
        bscore = d['bscore']
        obj = BootlegScore(bscore)
        obj.type = "Image"
        return obj
    

    def align_to_midi(self, ref, optimized = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the alignment between this query "Image" bootleg and the input "MIDI" reference bootleg score via Dynamic Time Warping (DTW)

        Params:
            ref (BootlegScore): the reference "MIDI" bootleg score
            optimized (bool): whether to use the optimized cython version of DTW

        Returns:
            (np.ndarray): the accumulated cost matrix computed by DTW, of dim = (n_query_frames x n_ref_frames);
                        each entry D[i, j] represents the minimum cost to align the first i frames of query with the first j frames of ref
            (np.ndarray): the optimal warping path, of dim = (n_steps x 2), represented as a sequence of (query_frame_index, ref_frame_index) pairs;
                        each pair shows which frame in the query aligns with which frame in the ref
            (float): the final cost of the alignment
        """
        if (not self.type == "Image") or (not ref.type == "MIDI"):
            print("ERROR: must call 'align_to_midi' method from an 'Image' bootleg, with a 'MIDI' bootleg input")
            return
        D, wp, end_cost = align.align_bootleg_scores(self.X, ref.X, ref.num_notes, BootlegScore.dtw_steps, BootlegScore.dtw_weights, optimized)
        self.aligned_to = ref
        self.cost_matr = D
        self.warping_path = wp
        return D, wp, end_cost

    def align_to_pdf(self, ref, optimized = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the alignment between this query "Image" bootleg and the input "PDF" reference bootleg score via Dynamic Time Warping (DTW)

        Params:
            ref (BootlegScore): the reference "PDF" bootleg score
            optimized (bool): whether to use the optimized cython version of DTW

        Returns:
            (np.ndarray): the accumulated cost matrix computed by DTW, of dim = (n_query_frames x n_ref_frames);
                        each entry D[i, j] represents the minimum cost to align the first i frames of query with the first j frames of ref
            (np.ndarray): the optimal warping path, of dim = (n_steps x 2), represented as a sequence of (query_frame_index, ref_frame_index) pairs;
                        each pair shows which frame in the query aligns with which frame in the ref
            (float): the final cost of the alignment    
        """
        if (not self.type == "Image") or (not ref.type == "Image"):
            print("ERROR: must call 'align_to_pdf' method from an 'Image' bootleg, with a 'PDF' bootleg input")
            return
        ref.num_notes = np.sum(ref.X, axis=0)
        D, wp, end_cost = align.align_bootleg_scores(self.X, ref.X, ref.num_notes, BootlegScore.dtw_steps, BootlegScore.dtw_weights, optimized)
        self.aligned_to = ref
        self.cost_matr = D
        self.warping_path = wp
        return D, wp, end_cost
    

    def align_to_query(self, query, optimized = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the alignment between this reference "MIDI" bootleg and the input "Image" query bootleg score via Dynamic Time Warping (DTW)

        Params:
            query (BootlegScore): the query "Image" bootleg score
            optimized (bool): whether to use the optimized cython version of DTW

        Returns:
            (np.ndarray): the accumulated cost matrix computed by DTW, of dim = (n_query_frames x n_ref_frames);
                        each entry D[i, j] represents the minimum cost to align the first i frames of query with the first j frames of ref
            (np.ndarray): the optimal warping path, of dim = (n_steps x 2), represented as a sequence of (query_frame_index, ref_frame_index) pairs;
                        each pair shows which frame in the query aligns with which frame in the ref
            (float): the final cost of the alignment
        """
        if (not self.type == "MIDI") or (not query.type == "Image"):
            print("ERROR: must call 'align_to_query' method from a 'MIDI' bootleg, with an 'Image' bootleg input")
            return
        D, wp, end_cost = align.align_bootleg_scores(query.X, self.X, self.num_notes, BootlegScore.dtw_steps, BootlegScore.dtw_weights, optimized)
        self.aligned_to = query
        self.cost_matr = D
        self.warping_path = wp
        return D, wp, end_cost
    

    def visualize_alignment(self):
        """
        If present, plot the last alignment computed between this bootleg score and another via DTW.
        """
        bs, D, wp = self.aligned_to, self.cost_matr, self.warping_path
        if bs is None or D is None or wp is None:
            print("No alignment computed for this BootlegScore object")
            return
        midi_times = self.times if self.type == 'MIDI' else bs.times
        img_file = self.img_file if self.type == 'Image' else bs.img_file
        matched_seg_time, _ = align.get_predicted_timestamps(wp, midi_times) # predicted timestamp in seconds for start and end of the matched segment in the MIDI
        ref_seg_times, ref_seg_cols = align.get_ground_truth_timestamps(img_file, midi_times) # real timestamps and corresponding bootleg columns from query info file
        seg_info = (matched_seg_time, ref_seg_times, ref_seg_cols)
        align.plot_alignment(D, wp, seg_info)


    def visualize_aligned_bootleg_scores(self):
        """
        If this bootleg score has been aligned to another via DTW, visualize them in the same plot (one on top of the other for each grand staff line)
        """
        bs, D, wp = self.aligned_to, self.cost_matr, self.warping_path
        if bs is None or D is None or wp is None:
            print("No alignment computed for this BootlegScore object")
            return
        if self.type == 'Image':
            align.visualize_aligned_bootleg_scores(self.X, bs.X, wp, BootlegScore.staff_lines)
        else:
            align.visualize_aligned_bootleg_scores(bs.X, self.X, wp, BootlegScore.staff_lines)

    
    @staticmethod
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
        return align.get_predicted_timestamps(wp, times)