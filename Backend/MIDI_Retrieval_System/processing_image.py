import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops
import matplotlib.patches as mpatches
from scipy.signal import convolve2d
from MIDI_Retrieval_System.musical_object_detection import MusicalObjectDetection
from sklearn.cluster import KMeans
import sys

class QueryProcessing:
    """
    Process a cellphone picture of some lines of sheet music via classical computer vision techniques and convert it to a booleg score.
    """

    ## Pre-processing parameters
    # Background lightning removal
    thumbnail_w = 100  
    thumbnail_h = 100
    thumbnail_filter_size = 5
    # Staff lines normalization
    est_line_sep_n_cols = 3
    est_line_sep_lower_bound = 25
    est_line_sep_upper_bound = 45
    est_line_sep_step = 1
    target_line_sep = 10.0

    ## Staffline Detection parameters
    max_delta_row_initial = 50
    min_num_staves = 2
    max_num_staves = 12
    min_stave_separation = 6 * target_line_sep
    max_delta_row_refined = 15

    ## Bootleg Score Generation parameters
    bootleg_repeat_notes = 2 
    bootleg_filler = 1
    # positions of staff lines for right hand and left hand staves
    staff_lines_RH = [7,9,11,13,15]
    staff_lines_LH = [13,15,17,19,21]
    staff_lines_both = [13,15,17,19,21,35,37,39,41,43]


    def __init__(self, image_file):
        """
        Initialize the QueryProcessing with the path of the picture to process.
        """
        # Carica così oppure passa direttamente l'immagine PIL
        if (isinstance(image_file, str)):
            self.image_file = image_file
            self.img = Image.open(image_file)
        elif (isinstance(image_file, Image.Image)):
            self.img = image_file

        # convert the image to grayscale via the PIL library 
        self.gray_img = self.img.convert('L') # 'L' stands for 'luminance'

        self.img_no_backgrd_noise = None
        self.pre_processed_img = None
        self.norm_inv_img = None

        # link to an object that takes care of detecting musical elements on sheet music
        self.object_detector = None
    
    def assign_detector(self) -> MusicalObjectDetection:
        """
        Create and assign a musical object detector for this QueryProcessing object 

        Returns:
            (MusicalObjectDetection): the created detector
        """
        det = MusicalObjectDetection(self, self.pre_process_image(), self.normalize_pre_processed_image())
        self.object_detector = det
        return det
    
    @staticmethod
    def display_img(img: Image.Image, cmap: str = 'gray'):
        """
        Display on screen a PIL image with the specified colormap
        """
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def normalize_and_invert_image(img) -> np.ndarray:
        """
        Normalize the input image to have values between 0 and 1 and then invert them.

        Returns:
            (np.array): normalized and inverted image
        """
        return 1 - np.array(img) / 255.0 

    @staticmethod
    def show_grayscale_image(img: np.ndarray, fig_size: tuple[int, int] = (10, 10), max_val: int = 1, inverted: bool = True):
        """
        Show a gray scale image on screen.

        Params:
            img (np.ndarray): the image to show
            fig_size (tuple[int, int]): size of the plot.
            max_val (int): value for inverting the black and white on the image.
            inverted (bool): show the inverted colors image or not.
        """
        plt.figure(figsize=fig_size)
        if inverted:
            plt.imshow(max_val - img, cmap = 'gray')
        else:
            plt.imshow(img, cmap = 'gray')
        plt.show()

    @staticmethod
    def show_color_image(img, fig_size = (10, 10)):
        """
        Show an RGB image on screen.

        Params:
            img (np.ndarray): the image to show
            fig_size (tuple[int, int]): size of the plot.
        """
        plt.figure(figsize=fig_size)
        plt.imshow(img)
        plt.show()
    
    @staticmethod
    def show_img_with_bound_boxes(img, bboxes, fig_size = (10,10)):
        """
        Show an image with red bounding boxes on it on screen.

        Params:
            img (np.ndarray): the image to show
            bboxes (list[tuple[int,int,int,int]]): list of bounding boxes to draw
            fig_size (tuple[int, int]): size of the plot.
        """
        _, ax = plt.subplots(figsize=fig_size)
        ax.imshow(img)

        for (minr, minc, maxr, maxc) in bboxes:
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    def normalize_pre_processed_image(self) -> np.ndarray:
        """
        Normalize and invert the pre-processed image

        Returns:
            (np.ndarray) the pre-processed image normalized to be a binary image
        """
        if not self.pre_processed_img: self.pre_process_image()
        self.norm_inv_img = QueryProcessing.normalize_and_invert_image(self.pre_processed_img)
        return self.norm_inv_img       

    def pre_process_image(self, verbose: bool = False) -> np.ndarray:
        """
        Pre-process a picture of sheet music by removing noise caused by background lightning conditions 
        and resizing the image in order to have the desired amount of pixels in the spacing between adjacent staff lines

        Returns:
            (np.ndarray) the pre-processed image
        """

        if self.pre_processed_img:
            return self.pre_processed_img

        # remove background lightning to reduce noise
        self.img_no_backgrd_noise = self.remove_background_lightning()
        # normalize the spacing between staff lines by resizing the image
        estimated_line_sep, _ = self.estimate_line_sep(QueryProcessing.est_line_sep_n_cols, 
                                                    QueryProcessing.est_line_sep_lower_bound, 
                                                    QueryProcessing.est_line_sep_upper_bound, 
                                                    QueryProcessing.est_line_sep_step)
        target_h, target_w = self.calculate_resized_dimensions(estimated_line_sep, QueryProcessing.target_line_sep)
        self.pre_processed_img = self.img_no_backgrd_noise.resize((target_w, target_h))

        if verbose:
            print(f'Target Height: {target_h}, Target Width: {target_w}')
            print(f'Original Height: {self.gray_img.height}, Original Width: {self.gray_img.width}')

        return self.pre_processed_img


    def remove_background_lightning(self, filt_sz: int = thumbnail_filter_size, thumbnail_w: int = thumbnail_w, thumbnail_h: int = thumbnail_h) -> Image.Image:
        """
        Subtract a blurred version of the grayscale image from the original one 
        in order to remove background lightning and reduce noise
        """
        tiny_img = self.gray_img.copy()
        tiny_img.thumbnail([thumbnail_w, thumbnail_h]) # reduce image dimensions to lower computational cost
        shadows = tiny_img.filter(ImageFilter.GaussianBlur(filt_sz)).resize(self.gray_img.size)
        result = ImageChops.invert(ImageChops.subtract(shadows, self.gray_img))
        return result


    @staticmethod 
    def get_staff_lines_filter(line_sep) -> np.ndarray:
        """
        Create a periodic array that represents a filter to detect staff lines at a certain distance from each other
        (the filter will have positive spikes around the positions where staff lines are expected
         and negative spikes at regular intervals where spaces between staff lines are expected)
        """

        filter = np.zeros(int(np.round(line_sep * 5)))
        
        # positive spikes
        for i in range(5):
            offset = int(np.round(.5*line_sep + i*line_sep))
            filter[offset-1:offset+2] = 1.0 # the positive spike is 3-pixels wide and is centered around the expected position of a staff line
        
        # negative spikes
        for i in range(6):
            center = int(np.round(i*line_sep))
            startIdx = max(center - 1, 0)
            endIdx = min(center + 2, len(filter))
            filter[startIdx:endIdx] = -1.0 # the negative spike is 3-pixels wide and is centered around the expected position of a space
            
        return filter
    

    def estimate_line_sep(self, n_cols: int, low_bound: int, up_bound: int, step: int) -> tuple[int, np.ndarray]:
        """
        Estimate the spacing between staff lines by using multiple filters for different candidate spacings
        and considering the one that gives the highest response

        Params:
            n_cols (int): the number of columns to divide the img into
            low_bound (int): lower bound for the line separation values to try
            up_bound (int): upper bound for the line separation values to try
            step (int): step for the line separation values to try
        
        Returns:
            (int) the line separation value with the highest score
            (np.ndarray) the scores for all line separation values
        """
    
        # break image into columns, calculate row medians for inner columns (exclude outermost columns)
        if not self.img_no_backgrd_noise:
            self.img_no_backgrd_noise = self.remove_background_lightning()
        img = 255 - np.array(self.img_no_backgrd_noise) # invert colors so that staff lines become bright
        img_h, img_w = img.shape
        row_medians = np.zeros((img_h, n_cols))
        col_w = img_w // (n_cols + 2)
        for i in range(n_cols):
            # for each column, the median pixel value is computed for each row
            row_medians[:,i] = np.median(img[:,(i+1)*col_w:(i+2)*col_w], axis=1)
        
        # apply filters
        line_seps = np.arange(low_bound, up_bound, step)
        responses = np.zeros((len(line_seps), img_h, n_cols))
        for i, line_sep in enumerate(line_seps):
            filt = QueryProcessing.get_staff_lines_filter(line_sep).reshape((-1,1)) # reshape the filter into a column vector
            responses[i,:,:] = convolve2d(row_medians, filt, mode = 'same')
        
        # find filter with strongest response
        # for each candidate line_sep, the maximum response for each row is summed across all columns to compute a score
        scores = np.sum(np.max(responses, axis=1), axis=1)
        estimated_line_sep = line_seps[np.argmax(scores)]
        
        return estimated_line_sep, scores
    

    def calculate_resized_dimensions(self, estimated_line_sep, desired_line_sep) -> tuple[int, int]:
        """
        Get the dimensions of the image resized to have the spacing between adjacent staff lines equal to the desired quantity
        """
        if not self.img_no_backgrd_noise:
            self.img_no_backgrd_noise = self.remove_background_lightning()
        cur_h, cur_w = self.img_no_backgrd_noise.height, self.img_no_backgrd_noise.width
        scale_factor = 1.0 * desired_line_sep / estimated_line_sep
        target_h = int(cur_h * scale_factor)
        target_w = int(cur_w * scale_factor)
        return target_h, target_w
    

    ########################################## QUERY BOOTLEG PROJECTION #################################################

    @staticmethod
    def estimate_staff_line_locs(featmap: np.ndarray, notehead_locs: list[tuple[int,int,int,int]], stave_lens: np.ndarray, col_width: int, 
                                 delta_row_max: int, global_offset: int | list = 0) -> tuple[list[tuple[int, int, int, int, int]], int]:
        """
        Estimate the position of staff lines locally in context regions around noteheads.

        Params:
            featmap (np.ndarray): the staff feature map (of dimensions K x H x C, where K = number of comb filters, H = height of image in pixels, C = number of columns)
            notehead_locs (list[tuple[int,int,int,int]]): list of bounding boxes of single noteheads 
                                                          (each bounding box contains row minimum, column minimum, row maximum, column maximum)
            stave_lens (np.ndarray): an array containing the staff lenghts for each different considered spacing between staff lines
            col_width (int): whe width of each column in the feature map, used to determine which column a point falls into
            delta_row_max (int): maximum row offset around a point to consider when searching for the staff line in the feature map
            global_offset (int or list): an offset applied to the row range when searching for staff lines; if a scalar is passed, it is applied uniformly; 
                                         otherwise, a list allows specifying unique offsets for each notehead location

        Returns:
            (list[tuple[int, int, int, int, int]]): the list of predictions for the locations of the staff lines
                                                    (each prediction contains row start, row end, column and row of local region, index of filter for spacing)
            (int): estimated staff line filter length (computed as the median of all staff lengths)
        """
        preds = []
        if np.isscalar(global_offset):
            global_offset = [global_offset] * len(notehead_locs) # convert a single value to a list

        for i, nh_loc in enumerate(notehead_locs):
            r = int(np.round(nh_loc[0]))
            c = int(np.round(nh_loc[1]))
            r_upper = min(r + delta_row_max + 1 + global_offset[i], featmap.shape[1])
            r_lower = max(r - delta_row_max + global_offset[i], 0)
            featmap_idx = c // col_width
            local_region = np.squeeze(featmap[:, r_lower:r_upper, featmap_idx])
            # take the peak in the local region as the most likely staff line
            spacing_idx, row_offset = np.unravel_index(local_region.argmax(), local_region.shape) # convert index for flattened array in an index for a tensor
            r_start = r_lower + row_offset
            r_end = r_start + stave_lens[spacing_idx] - 1
            preds.append((r_start, r_end, c, r, spacing_idx))

        staff_filter_len = int(np.round(np.median([stave_lens[tup[4]] for tup in preds]))) # estimated staff length
        return preds, staff_filter_len
    
    @staticmethod
    def visualize_pred_staff_lines(preds: list[tuple[int, int, int, int, int]], img: np.ndarray, fig_sz: tuple[int, int] = (15,15)):
        """
        Show the staff lines in the list in input on the given image.

        Params:
            preds (list[tuple[int, int, int, int, int]]): the list of predictions for the locations of the staff lines
                                                          (each prediction contains row start, row end, column and row of local region, index of filter for spacing)
            img (np.ndarray): the image to show staff lines on
            fig_sz (tuple[int, int]): size of the figure
        """

        plt.figure(figsize=fig_sz)
        plt.imshow(1 - img, cmap = 'gray')
        rows1 = np.array([pred[0] for pred in preds]) # top staff line
        rows2 = np.array([pred[1] for pred in preds]) # bottom staff line
        cols = np.array([pred[2] for pred in preds]) # notehead col
        rows3 = np.array([pred[3] for pred in preds]) # notehead row
        plt.scatter(cols, rows1, c = 'r', s = 3)
        plt.scatter(cols, rows2, c = 'b', s = 3)
        plt.scatter(cols, rows3, c = 'y', s = 3)
        plt.show()

    @staticmethod
    def estimate_staff_midpoints(preds: list[tuple[int, int, int, int, int]], clusters_min: int, clusters_max: int, threshold: int) -> np.ndarray:
        """
        Estimate vertical staff midpoints locations globally by clustering local estimations with k-means,
        increasing the number of clusters until the minimum distance between two centroids falls below the threshold

        Params:
            preds (list[tuple[int, int, int, int, int]]): the list of predictions for the locations of the staff lines
                                                          (each prediction contains row start, row end, column and row of local region, index of filter for spacing)
            clusters_min (int): lower bound for the number of clusters
            clusters_max (int): upper bound for the number of clusters
            threshold (int): minimum distance between two staves

        Returns:
            (np.ndarray): a sorted array of estimated staff line midpoints based on the best clustering model
        """
        r = np.array([.5*(tup[0] + tup[1]) for tup in preds]) # midpts of estimated stave locations
        models = []
        for num_clusters in range(clusters_min, clusters_max + 1):
            kmeans = KMeans(n_clusters=num_clusters, n_init=1, random_state = 0).fit(r.reshape(-1,1))
            sorted_list = np.array(sorted(np.squeeze(kmeans.cluster_centers_)))
            mindiff = np.min(sorted_list[1:] - sorted_list[0:-1])
            if num_clusters > clusters_min and mindiff < threshold:
                break
            models.append(kmeans)
        
        return np.sort(np.squeeze(models[-1].cluster_centers_))
    
    @staticmethod
    def visualize_staff_midpoint_clustering(preds: list[tuple[int, int, int, int, int]], centers: np.ndarray):
        """
        Visually represent the clustering process for vertical staff line midpoints
        (the number of cluster centers estimates the number of staves in an image).

        Params:
            preds (list[tuple[int, int, int, int, int]]): the list of predictions for the locations of the staff lines
                                                          (each prediction contains row start, row end, column and row of local region, index of filter for spacing)
            centers (np.ndarray): a sorted array of estimated staff line midpoints based on the best clustering model
        """
        r = np.array([.5*(tup[0] + tup[1]) for tup in preds]) # midpts of estimated stave locations
        y_values = np.random.uniform(low=0.4, high=0.6, size=len(r))
        plt.plot(r, y_values, '.', label='Predicted Midpoints')
        for center in centers:
            plt.axvline(x=center, color='r')
        plt.xlabel("Estimated Staff Line Midpoints")
        plt.ylabel("Random Spread (for visualization)")
        plt.title("Staff Line Midpoint Clustering")
        plt.legend()
        plt.show()

    @staticmethod
    def assign_noteheads_to_staves(nh_locs: list[tuple[int,int,int,int]], stave_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Assign noteheads to the nearest staff based on the vertical distance between their row locations and the centers of the staves

        Params:
            nh_locs (list[tuple[int,int]]): list of center coordinates (row, col) of single noteheads 
            stave_centers (np.ndarray): a 1D array of vertical positions representing the centers of the staves

        Returns:
            (np.ndarray): array of indices indicating which staff each notehead is closest to
            (np.ndarray): array of vertical offsets (row difference) between each notehead and the center of its assigned staff.
        """
        # matrix where each row corresponds to the row positions of all noteheads, repeated for the number of staves
        nh_rows = np.matlib.repmat([tup[0] for tup in nh_locs], len(stave_centers), 1) # len(centers) x len(nh_locs)
        # matrix where each column corresponds to the vertical positions of all staves, repeated for the number of noteheads
        centers = np.matlib.repmat(stave_centers.reshape((-1,1)), 1, len(nh_locs)) # len(centers) x len(nh_locs)
        stave_idxs = np.argmin(np.abs(nh_rows - centers), axis=0)
        offsets = stave_centers[stave_idxs] - nh_rows[0,:] # row offset between note and staff midpoint
        return stave_idxs, offsets
    
    @staticmethod
    def visualize_clusters(img: np.ndarray, nhlocs: list[tuple[int,int,int,int]], clusters: np.ndarray, fig_sz: tuple[int,int] = (10,10)):
        """
        Visualize notehead clusters on the input image

        Params:
            img (np.ndarray): input grayscale image where noteheads are located
            nh_locs (list[tuple[int,int]]): list of center coordinates (row, col) of single noteheads 
            clusters(np.ndarray): array of indices indicating which staff each notehead is closest to
            fig_sz (tuple[int, int]): figure size
        """
        plt.figure(figsize=fig_sz)
        plt.imshow(1 - img, cmap = 'gray')
        rows = np.array([tup[0] for tup in nhlocs])
        cols = np.array([tup[1] for tup in nhlocs])
        plt.scatter(cols, rows, c=clusters) # assign colors to points based on their cluster assignment
        for i in range(len(clusters)):
            plt.text(cols[i], rows[i] - 15, str(clusters[i]), fontsize = 12, color='red') # annotate each point with cluster label
        plt.xlabel("Column Position")
        plt.ylabel("Row Position")
        plt.title("Notehead Clusters")
        plt.show()
    
    @staticmethod
    def estimate_note_labels(preds: list[tuple[int, int, int, int, int]]) -> list[int]:
        """
        Estimate notes positions on the musical staff based on the vertical alignment of noteheads with respect to their corresponding staff's midpoints
        
        Params:
            preds (list[tuple[int, int, int, int, int]]): the list of predictions for the locations of the staff lines
                                                          (each prediction contains row start, row end, column and row of local region, index of filter for spacing)

        Returns:
            (list[int]): list of predicted note positions on the staff
        """
        nh_vals = [] # estimated note labels
        for _, (r_start, r_end, _, r, _) in enumerate(preds):
            # if a stave has height L, there are 8 stave locations in (L-1) pixel rows
            stave_midpt = .5 * (r_start + r_end)
            # relative location of the note w. r. t. the staff midpoint
            note_stave_loc = -1.0 * (r - stave_midpt) * 8 / (r_end - r_start) # negative values represent notes below the midpoint
            nh_val = int(np.round(note_stave_loc))
            nh_vals.append(nh_val)
        return nh_vals
    
    @staticmethod
    def visualize_note_labels(img: np.ndarray, vals: list[int], nh_locs: list[tuple[int,int,int,int]], fig_sz: tuple[int, int] = (10,10)):
        """
        Display the input image with noteheads labelled with their positions on the staff.

        Params:
            img (np.ndarray): grayscale image to display
            vals (list[int]): list of note labels corresponding to the noteheads' positions
            nh_locs (list[tuple[int,int]]): list of center coordinates (row, col) of single noteheads 
            fig_sz (tuple[int, int]): size of the figure to display
        """
        plt.figure(figsize=fig_sz)
        plt.imshow(1 - img, cmap = 'gray')
        rows = np.array([loc[0] for loc in nh_locs])
        cols = np.array([loc[1] for loc in nh_locs])
        plt.scatter(cols, rows, color='blue') # show notehead positions with blue dots
        for i in range(len(rows)):
            plt.text(cols[i], rows[i] - 15, str(vals[i]), fontsize = 12, color='red') # annotate with note positions in red
        plt.xlabel("Column Position")
        plt.ylabel("Row Position")
        plt.title("Annotated Noteheads")
        plt.show()

    @staticmethod
    def determine_stave_grouping(stave_mid_pts: np.ndarray, 
                                 bar_lines: np.ndarray) -> tuple[dict[int, int], tuple[float, float, list[float], list[float]]]:
        """
        Group the staves into right-hand/left-hand pairs, based on staves midpoints and bar line features:
        the right hand and left hand staves are connected by bar lines.

        This function attempts to group staves into two configurations (A and B):
        - Grouping A: Pairs consecutive staves (0-1, 2-3, 4-5, ...).
        - Grouping B: Pairs alternate staves starting from index 1 (1-2, 3-4, 5-6, ...).

        It evaluates the two groupings based on the median sum of bar line features
        within the respective groups and selects the grouping with higher "evidence".

        Params:
            stave_mid_pts (np.ndarray): array of midpoints representing the center of stave positions
            bar_lines (np.ndarray): 2D array where each row corresponds to features of bar lines for a stave row.

        Returns:
            dict[int, int]: dictionary mapping each stave index to -1 if unpaired, to itself if paired in Grouping A, to itself-1 if paired in Grouping B
            (float) evidence (median of row sums of bar line features) for Grouping A
            (float) evidence for Grouping B,
            (list[float]) bar line feature sums for elements in Grouping A
            (list[float]) bar line feature sums for elements in Grouping B
        """
        N = len(stave_mid_pts) # number of staves
        row_sums = np.sum(bar_lines, axis=1) # sum of bar lines (vertical lines) features for each row

        # grouping A: 0-1, 2-3, 4-5, ...
        elems_A = []
        map_A = {}
        for stave_idx in np.arange(0, N, 2):
            if stave_idx+1 < N:
                row_start = int(stave_mid_pts[stave_idx])       # consider a vertical region around the stave pairing
                row_end = int(stave_mid_pts[stave_idx+1]) + 1
                elems_A.extend(row_sums[row_start:row_end])
                map_A[stave_idx] = stave_idx
                map_A[stave_idx+1] = stave_idx + 1
            else:
                map_A[stave_idx] = -1 # unpaired stave

        # grouping B: 1-2, 3-4, 5-6, ...
        elems_B = []
        map_B = {}
        map_B[0] = -1
        for stave_idx in np.arange(1, N, 2):
            if stave_idx+1 < N:
                row_start = int(stave_mid_pts[stave_idx])
                row_end = int(stave_mid_pts[stave_idx+1]) + 1
                elems_B.extend(row_sums[row_start:row_end])
                map_B[stave_idx] = stave_idx - 1
                map_B[stave_idx + 1] = stave_idx
            else:
                map_B[stave_idx] = -1

        if N > 2:
            evidence_A = np.median(elems_A)
            evidence_B = np.median(elems_B)
            if evidence_A > evidence_B:
                mapping = map_A
            else:
                mapping = map_B
        else:
            evidence_A = np.median(elems_A)
            evidence_B = 0
            mapping = map_A

        return mapping, (evidence_A, evidence_B, elems_A, elems_B)
    
    @staticmethod
    def cluster_noteheads(stave_ids: list[int], mapping: dict[int, int]) -> tuple[list[int], list[tuple[int,int]]]:
        """
        Compute information useful to cluster noteheads based on staves.

        Params:
            stave_ids (list[int]): list of stave index for each notehead
            mapping (dict[int, int]): dictionary mapping each stave index to -1 if unpaired, to itself if paired in Grouping A, to itself-1 if paired in Grouping B
        Returns:
            (list[int]): list of staff (cluster) ids
            (list[tuple[int,int]]): list of staves (clusters) pairings
        """
        cluster_ids = [mapping[stave_idx] for stave_idx in stave_ids]
        max_cluster_idx = np.max(np.array(cluster_ids))
        cluster_pairs = []
        for i in range(0, max_cluster_idx, 2):
            cluster_pairs.append((i,i+1))
        return cluster_ids, cluster_pairs
    

    ########################################## BOOTLEG SCORE GENERATION #################################################

    @staticmethod
    def collapse_simultaneous_events(notes: tuple[int, int, int, int], min_col_diff: int) -> list[tuple[list[int], list[int], list[int], list[int]]]:
        """
        Group together noteheads that are sufficiently close in time

        Params:
            notes (tuple[int, int, int, int]): information on noteheads including row, column, value and cluster to which it belongs
            min_col_diff (int): threshold for collapsing simultaneous events

        Returns:
            (list[tuple[list[int], list[int], list[int], list[int]]): list of note events, each represented by a list of rows, columns, values and clusters of the component noteheads
        """
        assigned = np.zeros(len(notes), dtype=bool)
        events = [] # list of simultaneous note events
        for i, (row, col, val, cluster) in enumerate(notes):
            if assigned[i]: # has already been assigned
                continue
            rows = [row] # new event
            cols = [col]
            vals = [val]
            clusters = [cluster]
            assigned[i] = True
            for j in range(i+1, len(notes)):
                nrow, ncol, nval, ncluster = notes[j]
                if ncol - col < min_col_diff: # assign to same event if close
                    rows.append(nrow)
                    cols.append(ncol)
                    vals.append(nval)
                    clusters.append(ncluster)
                    assigned[j] = True
                else:
                    break
            events.append((rows, cols, vals, clusters))

        assert(np.all(assigned))
        return events
    
    @staticmethod
    def get_notehead_placement(vals: list[int], clusters: list[int], r_dim: int, l_dim: int, cluster_RH: int, cluster_LH: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Maps notehead placements to right-hand (RH) and left-hand (LH) vectors based on clusters.

        Args:
            vals (list[int]): list of notehead positions relative to a base index (e.g., -11 to +11)
            clusters (list[int]): list of cluster IDs corresponding to each notehead (each ID indicates left or right hand)
            r_dim (int): dimension of the right-hand vector
            l_dim (int): dimension of the left-hand vector
            cluster_RH (int): Cluster ID for right-hand noteheads
            cluster_LH (int): Cluster ID for left-hand noteheads

        Returns:
            (np.ndarray): binary vector indicating placements for right-hand noteheads
            (np.ndarray): binary vector indicating placements for left-hand noteheads

        Raises:
            AssertionError: If `cluster_LH` is not `cluster_RH + 1`.
            SystemExit: If an invalid cluster is encountered.
        """
        rhvec = np.zeros((r_dim, 1))
        lhvec = np.zeros((l_dim, 1))
        assert(cluster_LH == cluster_RH + 1)
        for (val, cluster) in zip(vals, clusters):
            if cluster == cluster_RH:
                idx = val + 11
                if idx >= 0 and idx < r_dim:
                    rhvec[idx, 0] = 1
            elif cluster == cluster_LH:
                idx = val + 17
                if idx >= 0 and idx < l_dim:
                    lhvec[idx, 0] = 1
            else:
                print("Invalid cluster: {} (LH {}, RH {})".format(cluster, cluster_LH, cluster_RH))
                sys.exit(1)
        return rhvec, lhvec
    
    @staticmethod
    def construct_bootleg_score(note_events: list[tuple[list[int], list[int], list[int], list[int]]], cluster_id_RH: int, cluster_id_LH: int, 
                                repeat_notes: int = 1, filler: int = 1) -> tuple[np.ndarray, list[int], tuple[np.ndarray, list[int]], tuple[np.ndarray, list[int]]]:
        """
        Group note events into a matrix of right and left hand note placements for each event

        Params:
            note_events (list[tuple[list[int], list[int], list[int], list[int]]]): list of note events, 
                                                                                  each represented by a list of rows, columns, values and clusters of the component noteheads
            cluster_id_RH (int): id of the cluster for the right hand
            cluster_id_LH (int): id of the cluster for the left hand
            repeat_notes (int): how many times to repeat each note event
            filler (int): how many filler colummns to insert between note events

        Returns:
            (np.ndarray): 2D array with note placements on right and left hand staves (dim: (lh_dim + rh_dim) x number of note events)
            (list[int]): list of event indexes to indicate which event each filler column is related to
            (tuple[np.ndarray, list[int]]): note placements for right hand and right hand staff lines
            (tuple[np.ndarray, list[int]]): note placements for left hand and right left staff lines
        
        """
       
        rh_dim = 34 # E3 to C8 (inclusive)
        lh_dim = 28 # A1 to G4 (inclusive)
        rh = [] # list of arrays of size rh_dim
        lh = [] # list of arrays of size lh_dim
        event_idxs = [] # index of corresponding simultaneous note event
        for i, (rows, cols, vals, clusters) in enumerate(note_events):

            # insert empty filler columns between note events
            if i > 0:
                for _ in range(filler):
                    rh.append(np.zeros((rh_dim,1)))
                    lh.append(np.zeros((lh_dim,1)))
                    event_idxs.append(i-1) # assign filler to previous event

            # insert note events columns
            rhvec, lhvec = QueryProcessing.get_notehead_placement(vals, clusters, rh_dim, lh_dim, cluster_id_RH, cluster_id_LH)
            for _ in range(repeat_notes):
                rh.append(rhvec)
                lh.append(lhvec)
                event_idxs.append(i)
        rh = np.squeeze(np.array(rh)).reshape((-1, rh_dim)).T # reshape handles case when len(rh) == 1
        lh = np.squeeze(np.array(lh)).reshape((-1, lh_dim)).T
        both = np.vstack((lh, rh))
        return both, event_idxs, (rh, QueryProcessing.staff_lines_RH), (lh, QueryProcessing.staff_lines_LH)

    @staticmethod
    def generate_single_bootleg_line(notes_data: tuple[int, int, int, int], cluster_R: int, cluster_L: int, 
                                     min_col_diff: int, repeat_notes: int = 1, filler: int = 1) -> tuple[np.ndarray, list[tuple[list[int], list[int], list[int], list[int]]], list[int]]:
        """
        Generate the bootleg score corresponding to a single line of sheet music (including both left and right hand staves)

        Params:
            notes_data (tuple[int, int, int, int]): information on noteheads including row, column, value and cluster to which it belongs
            cluster_R (int): id of the cluster for the right hand
            cluster_L (int): id of the cluster for the left hand
            min_col_diff (int): threshold for collapsing simultaneous events
            repeat_notes (int): how many times to repeat each note event
            filler (int): how many filler colummns to insert between note events

        Returns:
            (np.ndarray): the bootleg score line
            (list[tuple[list[int], list[int], list[int], list[int]]]): list of note events, each represented by a list of rows, columns, values and clusters of the component noteheads
            (list[int]): list of event indexes to indicate which event each filler column is related to
        """
        notes = [tup for tup in notes_data if tup[3] == cluster_R or tup[3] == cluster_L]
        notes = sorted(notes, key = lambda tup: (tup[1], tup[0])) # sort by column, then row
        collapsed = QueryProcessing.collapse_simultaneous_events(notes, min_col_diff) # list of (rows, cols, vals, clusters)
        bscore, event_idxs, _, _ = QueryProcessing.construct_bootleg_score(collapsed, cluster_R, cluster_L, repeat_notes, filler)
        return bscore, collapsed, event_idxs
    
    @staticmethod
    def visualize_bootleg_score(bs: np.ndarray, lines: list[int], fig_sz: tuple[int, int] = (10,10)):
        """
        Visualize on screen a bootleg score with staff lines.

        Params:
            bs (np.ndarray): the bootleg score to visualize
            lines (list[int]): the positions of staff lines for both left and right hands
            fig_sz (tuple[int, int]): the size of the figure to show
        """
        plt.figure(figsize = fig_sz)
        plt.imshow(1 - bs, cmap = 'gray', origin = 'lower')
        for l in range(1, bs.shape[0], 2):
            plt.axhline(l, c = 'grey')
        for l in lines:
            plt.axhline(l, c = 'r')
        plt.show()
    
    @staticmethod
    def generate_query_bootleg_score(notes_data: tuple[int, int, int, int], pairings: list[tuple[int,int]], 
                                     min_col_diff: int = 10, repeat_notes: int = 1, 
                                     filler: int = 1) -> tuple[np.ndarray, list[tuple[list[int], list[int], list[int], list[int]]], list[int]]:
        """
        Generate the complete bootleg score of the query by its extracted notes and stave clusters information.

        Params:
            notes_data (tuple[int, int, int, int]): information on noteheads including row, column, value and cluster to which it belongs
            pairings (list[tuple[int,int]]): list of staves (clusters) pairings
            min_col_diff (int): threshold for collapsing simultaneous events
            repeat_notes (int): how many times to repeat each note event
            filler (int): how many filler colummns to insert between note events

        Returns:
            (np.ndarray): bootleg score resulting from the horizontal concatenation of single bootleg lines
            (list[tuple[list[int], list[int], list[int], list[int]]]): list of all note events, each represented by a list of rows, columns, values and clusters of the component noteheads
            (list[int]): list of event indexes to indicate which event each filler column is related to

        """
        all_scores = []
        all_events = []
        glob_idxs = []
        event_count = 0
        for i, (cluster_R, cluster_L) in enumerate(pairings):
            score, events, event_idxs = QueryProcessing.generate_single_bootleg_line(notes_data, cluster_R, cluster_L, min_col_diff, repeat_notes, filler)
            all_scores.append(score)
            all_events.extend(events)
            glob_idxs.extend([idx + event_count for idx in event_idxs])
            if i < len(pairings) - 1:
                all_scores.append(np.zeros((score.shape[0], filler))) # append filler columns between bootleg scores
                glob_idxs.extend([glob_idxs[-1]] * filler) # map filler columns to last event index
            event_count += len(events)
        panorama = np.hstack(all_scores)
        return panorama, all_events, glob_idxs
    
    @staticmethod
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
            QueryProcessing.visualize_bootleg_score(bs[:,start_col:end_col], lines)