import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops
import cv2
import matplotlib.patches as mpatches
from scipy.signal import convolve2d
from scipy.spatial import KDTree
import seaborn as sns
import pickle
import librosa as lb
import time
import cProfile
import os
import os.path
#import pyximport; pyximport.install()
import multiprocessing
from MIDI_Retrieval_System.musical_object_detection import MusicalObjectDetection

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


    ## Staffline Detection
    maxDeltaRowInitial = 50
    minNumStaves = 2
    maxNumStaves = 12
    minStaveSeparation = 6 * target_line_sep
    maxDeltaRowRefined = 15

    ## Group Staves
    morphFilterVertLineLength = 101
    morphFilterVertLineWidth = 7
    maxBarlineWidth = 15
    #maxBarlineLenFactor = .25

    ## Generate Bootleg Score
    bootlegRepeatNotes = 2 
    bootlegFiller = 1

    ## Alignment
    dtw_steps = [1,1,1,2,2,1] # dtw
    dtw_weights = [1,1,2]


    def __init__(self, image_file: str):
        """
        Initialize the QueryProcessing with the path of the picture to process.
        """
        self.image_file = image_file
        self.img = Image.open(image_file)

        # convert the image to grayscale via the PIL library 
        self.gray_img = self.img.convert('L') # 'L' stands for 'luminance'
        self.pre_processed_image = None

        # link to an object that takes care of detecting musical elements on sheet music
        self.object_detector = None

    
    @staticmethod
    def normalize_and_invert_image(img) -> np.ndarray:
        """
        Normalize the input image to have values between 0 and 1 and then invert them.

        Returns:
            (np.array): normalized and inverted image
        """
        return 1 - np.array(img) / 255.0
    

    @staticmethod
    def show_grayscale_image(img, fig_size = (10, 10), max_val = 1, inverted = True):
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


    def get_normalized_pre_processed_image(self) -> np.ndarray:
        """
        Returns:
            (np.ndarray) the pre-processed image normalized to be a binary image
        """
        if not self.pre_processed_image: self.pre_process_image()
        return QueryProcessing.normalize_and_invert_image(self.pre_processed_image)            


    def pre_process_image(self) -> np.ndarray:
        """
        Pre-process a picture of sheet music by removing noise caused by background lightning conditions 
        and resizing the image in order to have the desired amount of pixels in the spacing between adjacent staff lines

        Returns:
            (np.ndarray) the pre-processed image
        """
        # remove background lightning to reduce noise
        self.pre_processed_image = self.remove_background_lightning()
        # normalize the spacing between staff lines by resizing the image
        estimated_line_sep, _ = self.estimate_line_sep(QueryProcessing.est_line_sep_n_cols, 
                                                    QueryProcessing.est_line_sep_lower_bound, 
                                                    QueryProcessing.est_line_sep_upper_bound, 
                                                    QueryProcessing.est_line_sep_step)
        target_h, target_w = self.calculate_resized_dimensions(estimated_line_sep, QueryProcessing.target_line_sep)
        self.pre_processed_image = self.pre_processed_image.resize((target_w, target_h))
        return self.pre_processed_image


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
    

    def estimate_line_sep(self, n_cols, low_bound, up_bound, step) -> tuple[int, np.ndarray]:
        """
        Estimate the spacing between staff lines by using multiple filters for different candidate spacings
        and considering the one that gives the highest response
        """
    
        # break image into columns, calculate row medians for inner columns (exclude outermost columns)
        img = 255 - np.array(self.gray_img) # invert colors so that staff lines become bright
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
        cur_h, cur_w = self.gray_img.height, self.gray_img.width
        scale_factor = 1.0 * desired_line_sep / estimated_line_sep
        target_h = int(cur_h * scale_factor)
        target_w = int(cur_w * scale_factor)
        return target_h, target_w