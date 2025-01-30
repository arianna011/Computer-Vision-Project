import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage import filters, measure
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from sklearn.cluster import KMeans

class MusicalObjectDetection:
    """
    Compute information to predict the locations of musical objects in a normalized image of sheet music, such as staff lines, noteheads and bar lines.
    """

    # Staff Line Features
    morph_filter_rect_len = 41
    notebar_filter_len = 3
    notebar_removal = 0.9

    stave_feat_map_n_cols = 10
    stave_feat_map_lower_bound = 8.5
    stave_feat_map_upper_bound = 11.75
    stave_feat_map_step = 0.25

    # Notehead Detection
    morph_filter_circ_dilate = 5
    morph_filter_circ_erode = 5
    note_detect_min_area = 50  
    note_detect_max_area = 200  
    note_template_size = 21
    note_detect_tol_ratio = .4

    chord_min_h = 1.25
    chord_max_h = 4.25
    chord_min_w = .8
    chord_max_w = 2.25
    chord_min_area = 1.8
    chord_max_area = 4.5
    chord_min_notes = 2
    chord_max_notes = 4
    chord_specs = (chord_min_h, chord_max_h, chord_min_w, 
                   chord_max_w, chord_min_area, chord_max_area, 
                   chord_min_notes, chord_max_notes)
    
    # Bar Line Features
    morph_filter_bar_vert = 101
    morph_filter_bar_hor = 7
    max_barline_width = 15


    def __init__(self, qproc, img, norm_img):
        """
        Initialize the MusicalObjectDetection with the related QueryProcessing object, a grayscale pre-processed image and a normalized (binary) pre-processed image of sheet music.
        """
        self.qproc = qproc
        self.img = img
        self.norm_img = norm_img

        self.isol_staff_lines = None 
        self.notes_bboxes = []
        self.isol_bar_lines = None

    @staticmethod
    def morph_filter_rectangle(img: np.ndarray , kernel_h: int, kernel_w: int) -> np.ndarray:
        """
        Perform erosion and dilation (morphological opening operation) on the input binary image
        with a rectangular filter

        Params:
            img (np.ndarray): binary image
            kernel_h (int): filter height
            kernel_w (int): filter width

        Returns:
            (np.ndarray): result of the opening operation
        """
        kernel = np.ones((kernel_h, kernel_w), dtype = np.uint8)
        result = cv2.erode(img, kernel, iterations = 1)
        result = cv2.dilate(result, kernel, iterations = 1)
        return result
    
    @staticmethod
    def morph_filter_circle(img: np.ndarray, sz_dilate: int = 5, sz_erode: int = 0) -> np.ndarray:
        """
        Perform dilation and erosion (morphological closing operation) on the input binary image
        with a circular filter

        Params:
            img (np.ndarray): binary image
            sz_dilate (int): size of the filter for dilation
            sz_erode (int): size of the filter for erosion

        Returns:
            (np.ndarray): result of the closing operation
        """
        dilate_filt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz_dilate, sz_dilate))
        result = cv2.dilate(np.array(img), dilate_filt, iterations = 1)
        if sz_erode > 0:
            erode_filt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz_erode, sz_erode))
            result = cv2.erode(result, erode_filt, iterations = 1)
        return result


    ################################ STAFF LINES FEATURES ############################################

    def isolate_staff_lines(self, kernel_len: int, notbar_filter_len: int, notebar_removal: float) -> np.ndarray:
        """
        Keep only the staff lines in a binary image of sheet music 
        by first applying erosion and dilation with an horizontal morphological filters
        to filter out everything except horizontal lines
        and then subtracting notebars (they survirve the first filtering since they are horizontal too)

        Params:
            kernel_len (int): length of the filter used for erosion and dilation
            notbar_filter_len (int): lenght of the filter used to isolate notebars
            notebar_removal (): percentage that determines at which extent the notebars are subtracted (e.g. notebar_removal = 1 means the notebars are fully subtracted)

        Returns:
            (np.ndarray): binary image of sheet music with isolated staff lines
        """

        lines = self.morph_filter_rectangle(img = self.norm_img, kernel_h = 1, kernel_w = kernel_len) # isolate horizontal lines
        notebars_only = self.morph_filter_rectangle(img = lines, kernel_h = notbar_filter_len, kernel_w = 1) # isolate thick notebars
        result = np.clip(lines - notebar_removal * notebars_only, 0, None) # subtract out notebars and clip to ensure there are no negative values

        self.isol_staff_lines = result
        
        return result
    
    @staticmethod
    def get_comb_filter(line_sep: float) -> tuple[np.ndarray, int]:
        """
        Create a comb filter (an array with spikes in certain positions) to represent the staff lines spacing indicated by line_sep.
        Params:
            line_sep (float): spacing between adjacent staff lines
        Returns:
            (np.ndarray): comb filter
            (int): length of the stave with the chosen line separation
        """
        # generate comb filter of specified length
        # e.g. if length is 44, then spikes at indices 0, 11, 22, 33, 44
        # e.g. if length is 43, then spikes at 0 [1.0], 10 [.25], 11 [.75], 21 [.5], 22 [.5], 32 [.75], 33 [.25], 43 [1.0]
        stavelen = int(np.ceil(4 * line_sep)) + 1
        combfilt = np.zeros(stavelen)
        for i in range(5):
            idx = i * line_sep
            idx_below = int(idx)
            idx_above = idx_below + 1
            remainder = idx - idx_below
            combfilt[idx_below] = 1 - remainder
            if idx_above < stavelen:
                combfilt[idx_above] = remainder
        return combfilt, stavelen
    

    def compute_staff_feature_map(self, n_cols, lower_bound, upper_bound, step):
        """
        Compute the feature map of the musical staff in a cellphone image (needed since lines could be not perfectly straight in the image)
        by dividing the input image into a fixed number of columns,
        computing the row medians for each column and convolving them with different filters 
        that represent multiple possibile spacings between adjacent staff lines

        Params:
            n_cols (int): number of columns to divide the image into
            lower_bound (float): lower bound in the range of candidate spacings between staff lines
            upper_bound (float): upper bound in the range of candidate spacings between staff lines
            step (float): step between the values in the range of candidate spacings between staff lines

        Returns:
            (np.ndarray): the staff feature map (of dimensions K x H x C, where K = number of comb filters, H = height of image in pixels, C = number of columns)
            (np.ndarray): an array containing the filter size (equivalent to staff lenght) for each different spacing between staff lines
            (int): how wide each column is in the original image
        """

        # break image into columns, calculate row medians for each column
        img_h, img_w = self.isol_staff_lines.shape
        row_sums = np.zeros((img_h, n_cols))
        col_w = int(np.ceil(img_w/n_cols))
        for i in range(n_cols):
            start_col = i * col_w
            end_col = min((i+1)*col_w, img_w)
            row_sums[:,i] = np.sum(self.isol_staff_lines[:,start_col:end_col], axis=1) # ? here it actually computes row sums
        
        # apply comb filters
        line_seps = np.arange(lower_bound, upper_bound, step) # candidate values for staff lines separation
        max_filt_size = int(np.ceil(4 * line_seps[-1])) + 1
        featmap = np.zeros((len(line_seps), img_h - max_filt_size + 1, n_cols))  
        stave_lens = np.zeros(len(line_seps), dtype=int)

        for i, line_sep in enumerate(line_seps):
            filt, stave_len = MusicalObjectDetection.get_comb_filter(line_sep)
            padded = np.zeros((max_filt_size, 1))
            padded[0:len(filt),:] = filt.reshape((-1,1)) # pad with 0s so that each filter has size max_filt_sizenp
            featmap[i,:,:] = convolve2d(row_sums, np.flipud(np.fliplr(padded)), mode = 'valid') # flip the filter horizontally and vertically before applying
            stave_lens[i] = stave_len
            
        return featmap, stave_lens, col_w
    

    ################################ NOTEHEAD DETECTION ############################################

    def detect_notehead_blobs(self, min_area: float = None, max_area: float = None, min_threshold: float = None, max_threshold: float = None, 
                            min_circularity: float = None, min_convexity: float = None, min_intertia_ratio: float = None) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """
        Use a simple blob detector to recognize noteheads, based on multiple optional factors such as area or circularity,
        and draw the found keypoints on the eroded and dilated pre-processed image of sheet music

        Params:
            img (np.ndarray): image to apply the blob detector on
            min_area (float): minimum area of the blobs to consider
            max_area (float): maximum area of the blobs to consider
            min_threshold (float): lower bound for thresholding the input image
            max_threshold (float): upper bound for thresholding the input image
            min_circularity (float): minimum similarity to a circle of the blobs to consider (between 0 and 1)
            min_convexity (float): miniimum convexity of the blobs to consider (between 0 and 1)
            min_intertia_ratio (float): minimum value of how elongated the blobs to consider are (between 0 and 1)

        Returns:
            (list[cv2.KeyPoint]): the keypoints found by the blob detector
            (np.ndarray): the image with drawn keypoints
                
        """

        img = MusicalObjectDetection.morph_filter_circle(self.img, MusicalObjectDetection.morph_filter_circ_dilate, MusicalObjectDetection.morph_filter_circ_erode)

        #img = np.array(self.img)
        cv2.imshow('image_before',img)
        cv2.waitKey(0)
        # define blob detector parameters
        params = cv2.SimpleBlobDetector_Params()

        # Filter by Area
        if min_area:
            params.minArea = min_area
            params.filterByArea = True
        if max_area:
            params.maxArea = max_area
            params.filterByArea = True

        # Change thresholds
        if min_threshold:
            params.minThreshold = min_threshold
        if max_threshold:
            params.maxThreshold = max_threshold

        # Filter by Circularity
        if min_circularity:
            params.filterByCircularity = True
            params.minCircularity = min_circularity

        # Filter by Convexity
        if min_convexity:
            params.filterByConvexity = True
            params.minConvexity = min_convexity

        # Filter by Inertia
        if min_intertia_ratio:
            params.filterByInertia = True
            params.minInertiaRatio = min_intertia_ratio

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(img)
        print(f'Number of keypoints detected: {len(keypoints)}')
        im_with_keypoints = cv2.drawKeypoints(np.array(img), keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image',im_with_keypoints)
        cv2.waitKey(0)  
        return keypoints, im_with_keypoints 
    

    def get_note_template(self, keypoints: list[cv2.KeyPoint], sz: int) -> tuple[np.ndarray, int]:
        """
        Compute an estimate (template) of how a notehead looks like in the normalized eroded and dilated pre-processed image
        by taking the average of cropped regions around detected noteheads

        Params:
            keypoints (list[cv2.KeyPoint]): the list of keypoints resulting from notehead detection on the image
            sz (int): the size of the template

        Returns:
            (np.ndarray): the notehead template
            (int): the number of crops used

        """
        img = MusicalObjectDetection.morph_filter_circle(self.img, MusicalObjectDetection.morph_filter_circ_dilate, MusicalObjectDetection.morph_filter_circ_erode)
        img = self.qproc.normalize_and_invert_image(img)

        template = np.zeros((sz,sz))
        L = (sz - 1)//2 # used to define a crop region
        num_crops = 0
        for k in keypoints:
            xloc = int(np.round(k.pt[0])) # col
            yloc = int(np.round(k.pt[1])) # row
            if xloc - L >= 0 and xloc + L + 1 <= img.shape[1] and yloc - L >= 0 and yloc + L + 1 <= img.shape[0]:
                crop = img[yloc-L:yloc+L+1,xloc-L:xloc+L+1]
                template += crop
                num_crops += 1
        if num_crops > 0:
            template = template / num_crops
        return template, num_crops

    @staticmethod
    def binarize_otsu(img: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Binarize the input grayscale image with the Otsu method (automatically determines an optimal global threshold value from the image histogram)

        Params:
            img (np.ndarray): the image to binarize
        Returns:
            (np.ndarray): the binarized image
            (float): threshold computed by Otsu algorithm
        """
        arr = np.array(img)
        thresh = filters.threshold_otsu(arr)
        binarized = arr > thresh
        return binarized, thresh    
    
    @staticmethod
    def get_note_template_specs(template: np.ndarray, thresh: float) -> tuple[int, int, int]:
        """
        Compute the dimensions and area of the largest connected component in a template image

        Params:
            template (np.ndarray): a binary or grayscale image that represents a notehead template
            thresh (float): threshold value used for binarizing the image
        Returns:
            (int): height of the largest component
            (int): width of the largest component
            (int): area of the largest component
        """
        binarized = template > thresh
        labels = measure.label(binarized) # image where each distinct connected component (foreground object) has a unique label
        max_h, max_w, max_area = (0, 0, 0)
        for region in regionprops(labels):
            cur_h = region.bbox[2] - region.bbox[0]
            cur_w = region.bbox[3] - region.bbox[1]
            cur_area = region.area
            if cur_area > max_area:
                max_area = cur_area
                max_h = cur_h
                max_w = cur_w
        return (max_h, max_w, max_area)
    

    @staticmethod
    def is_valid_notehead(region: object, tol_ratio: float, template_specs: tuple[int, int, int]) -> bool:
        """
        Determine if a given region is a valid notehead, based on its dimensions and aspect ratio compared to a reference template

        Params:
            region (object): input connected component, returned by the regionprops() function
            tol_ratio (float): factor of tolerance towards variations in size between the region and the template
            template_specs (tuple[int, int, int]): height, width and area of the notehead template

        Returns:
            (bool): true if the region is a valid notehead, false otherwise
        """
        template_h, template_w, template_area = template_specs
        max_ratio = 1 + tol_ratio # upper bound for valid area and dimensions
        min_ratio = 1 / (1 + tol_ratio) # lower bound for valid area and dimensions
        cur_h = region.bbox[2] - region.bbox[0]
        cur_w = region.bbox[3] - region.bbox[1]
        cur_area = region.area
        cur_ratio = 1.0 * cur_h / cur_w
        template_ratio = 1.0 * template_h / template_w

        valid_h = cur_h < template_h * max_ratio and cur_h > template_h * min_ratio
        valid_w = cur_w < template_w * max_ratio and cur_w > template_w * min_ratio
        valid_area = cur_area < template_area * max_ratio * max_ratio and cur_area > template_area * min_ratio * min_ratio
        valid_ratio = cur_ratio < template_ratio * max_ratio and cur_ratio > template_ratio * min_ratio
        
        return valid_h and valid_w and valid_ratio and valid_area
    
    @staticmethod
    def is_valid_chord_block(region: object, params: tuple[float, float, float, float, float, float, int, int], template_specs: tuple[int, int, int]) -> bool:
        """
        Determine if a given region is a valid chord block, based on some parameters indicating expected range of dimensions, area and number of notes
        with reference to a notehead template (a chord is a set of multiple noteheads)

        Params:
            region (object): input connected component, returned by the regionprops() function
            params (tuple[float, float, float, float, float, float, int, int]): min and max values for height, width, area and number of notes of the chord
            template_specs (tuple[int, int, int]): height, width and area of a notehead template

        Returns:
            (bool): true if the region is a valid chord block, false otherwise
        """
        template_h, template_w, template_area = template_specs
        min_h, max_h, min_w, max_w, min_area, max_area, min_notes, max_notes = params
        cur_h = region.bbox[2] - region.bbox[0]
        cur_w = region.bbox[3] - region.bbox[1]
        cur_area = region.area
        cur_notes = int(np.round(cur_area / template_area)) # estimate the number of notes in the chord based on a notehead template

        valid_h = cur_h >= min_h * template_h and cur_h <= max_h * template_h
        valid_w = cur_w >= min_w * template_w and cur_w <= max_w * template_w
        valid_area = cur_area >= min_area * template_area and cur_area <= max_area * template_area
        valid_notes = cur_notes >= min_notes and cur_notes <= max_notes

        return valid_h and valid_w and valid_area and valid_notes
    
    @staticmethod
    def extract_notes_from_chord_block(region: object, template_specs: tuple[int, int, int]) -> list[tuple[int,int,int,int]]:
        """
        Compute the bounding boxes of the single noteheads that compose a chord block
        by using K-means clustering of individual pixel coordinates in the connected component region

        Params:
            region (object): connected component representing a chord block
            template_specs (tuple[int, int, int]): height, width and area of a notehead template

        Returns:
            (list[tuple[int,int,int,int]]): list of bounding boxes of single noteheads 
                                            (each bounding box contains row minimum, column minimum, row maximum, column maximum)
        """
        template_h, template_w, template_area = template_specs
        num_notes = int(np.round(region.area / template_area))
        region_coords = np.array(region.coords)
        # use k-means to estimate note centers
        kmeans = KMeans(n_clusters=num_notes, n_init = 1, random_state = 0).fit(region_coords)
        bboxes = []
        for (r,c) in kmeans.cluster_centers_:
            rmin = int(np.round(r - template_h/2))
            rmax = int(np.round(r + template_h/2))
            cmin = int(np.round(c - template_w/2))
            cmax = int(np.round(c + template_w/2))
            bboxes.append((rmin, cmin, rmax, cmax))
        return bboxes

    def adaptive_notehead_detect(self, template: np.ndarray, note_tol_ratio: float, 
                                 chord_block_specs: tuple[float, float, float, float, float, float, int, int]) -> tuple[list[tuple[int,int,int,int]], np.ndarray]:
        """
        Detect noteheads in the normalized image, by considering both isolated noteheads and noteheads in chord blocks

        Params:
            template (np.ndarray): template of a notehead
            chord_block_specs (tuple[float, float, float, float, float, float, int, int]): min and max values for height, width, area and number of notes of a chord

        Returns:
            (list[tuple[int,int,int,int]]): list of bounding boxes of single noteheads 
                                          (each bounding box contains row minimum, column minimum, row maximum, column maximum)
            (np.ndarray): binarized version of the input image                           
        """
        img = MusicalObjectDetection.morph_filter_circle(self.img, MusicalObjectDetection.morph_filter_circ_dilate, MusicalObjectDetection.morph_filter_circ_erode)
        img = self.qproc.normalize_and_invert_image(img)

        binarized, thresh = MusicalObjectDetection.binarize_otsu(img)
        template_specs = MusicalObjectDetection.get_note_template_specs(template, thresh)
        labels = measure.label(binarized)
        notes = []
        if template.max() == 0: # no noteheads detected
            return notes, binarized
        for region in regionprops(labels):
            if MusicalObjectDetection.is_valid_notehead(region, note_tol_ratio, template_specs):
                notes.append(region.bbox)
            elif MusicalObjectDetection.is_valid_chord_block(region, chord_block_specs, template_specs):
                chord_notes = MusicalObjectDetection.extract_notes_from_chord_block(region, template_specs)
                notes.extend(chord_notes)
        self.notes_bboxes = notes
        return notes, binarized
    
    def get_notehead_info(self) -> tuple[list[tuple[float, float]], int, int]:
        """
        Compute information about the noteheads bounding boxes: 
        the center coordinates (row, column) of each notehead and the estimated average length and width of noteheads

        Returns:
            (list[tuple[float, float]]): a list of center coordinates of each notehead
            (int): rounded estimated average length of noteheads
            (int): rounded estimated average width of noteheads
                              
        """
        center_coords = [(.5*(bbox[0] + bbox[2]), .5*(bbox[1] + bbox[3])) for bbox in self.notes_bboxes]
        heights = [(bbox[2] - bbox[0]) for bbox in self.notes_bboxes]
        widths = [(bbox[3] - bbox[1]) for bbox in self.notes_bboxes]
        mean_heights = int(np.ceil(np.mean(heights)))
        mean_widths = int(np.ceil(np.mean(widths)))
        return center_coords, mean_heights, mean_widths
    
    
    ################################ BARLINE FEATURES ############################################
    
    def isolate_bar_lines(self, vert_filter: int, hor_filter: int, max_barline_width: int) -> np.ndarray:
        """
        Compute features about the location of bar lines in the normalized image,
        by first applying a dilation in the horizontal direction to expand bar lines that are not perfectly vertical,
        then performing erosion and dilation in the vertical direction to isolate bar lines,
        removing components that are too tick to be barlines

        Params:
            vert_filter (int): the dimension of the kernel for the vertical erosion and dilation
            hor_filter (int): the dimension of the kernel for the horizontal dilation

        Returns:
            (np.ndarray): a binary image with isolated bar lines

        """
        hor_kernel = np.ones((1, hor_filter), np.uint8) # dilate first to catch warped barlines
        vlines = cv2.dilate(self.norm_img, hor_kernel, iterations = 1)
        vlines = self.morph_filter_rectangle(vlines, vert_filter, 1) # then filter for tall vertical lines
        nonbarlines = self.morph_filter_rectangle(vlines, 1, max_barline_width) # extract elements that are too thick to be barlines
        vlines = np.clip(vlines - nonbarlines, 0, 1)
        self.isol_bar_lines = vlines
        return vlines