import numpy as np
import cv2
from scipy.signal import convolve2d

class MusicalObjectDetection:
    """
    Compute information to predict the locations of musical objects in a normalized image of sheet music, such as staff lines, noteheads and bar lines.
    """

    # Staff Line Features
    morph_filter_len = 41
    notebar_filter_len = 3
    notebar_removal = 0.9

    stave_feat_map_n_cols = 10
    stave_feat_map_lower_bound = 8.5
    stave_feat_map_upper_bound = 11.75
    stave_feat_map_step = 0.25


    def __init__(self, norm_img):
        """
        Initialize the MusicalObjectDetection with a normalized image of sheet music.
        """
        self.norm_img = norm_img


    def morph_filter_rectangle(self, img: np.ndarray , kernel_h: int, kernel_w: int) -> np.ndarray:
        """
        Perform erosion and dilation (morphological opening operation) on the input binary image

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


    def isolate_staff_lines(self, img: np.ndarray, kernel_len: int, notbar_filter_len: int, notebar_removal: float) -> np.ndarray:
        """
        Keep only the staff lines in a binary image of sheet music 
        by first applying erosion and dilation with an horizontal morphological filters
        to filter out everything except horizontal lines
        and then subtracting notebars (they survirve the first filtering since they are horizontal too)

        Params:
            img (np.ndarray): binary image of sheet music
            kernel_len (int): length of the filter used for erosion and dilation
            notbar_filter_len (int): lenght of the filter used to isolate notebars
            notebar_removal (): percentage that determines at which extent the notebars are subtracted (e.g. notebar_removal = 1 means the notebars are fully subtracted)

        Returns:
            (np.ndarray): binary image of sheet music with isolated staff lines
        """

        lines = self.morph_filter_rectangle(img = img, kernel_h = 1, kernel_w = kernel_len) # isolate horizontal lines
        notebars_only = self.morph_filter_rectangle(img = lines, kernel_h = notbar_filter_len, kernel_w = 1) # isolate thick notebars
        result = np.clip(lines - notebar_removal * notebars_only, 0, None) # subtract out notebars and clip to ensure there are no negative values
        
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
    
    def compute_staff_feature_map(self, img, n_cols, lower_bound, upper_bound, step):
        """
        Compute the feature map of the musical staff in a cellphone image (needed since lines could be not perfectly straight in the image)
        by dividing the input image into a fixed number of columns,
        computing the row medians for each column and convolving them with different filters 
        that represent multiple possibile spacings between adjacent staff lines

        Params:
            img (np.ndarray): input image containing isolated staff lines
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
        img_h, img_w = img.shape
        row_sums = np.zeros((img_h, n_cols))
        col_w = int(np.ceil(img_w/n_cols))
        for i in range(n_cols):
            start_col = i * col_w
            end_col = min((i+1)*col_w, img_w)
            row_sums[:,i] = np.sum(img[:,start_col:end_col], axis=1) # ? here it actually computes row sums
        
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