import numpy as np
import matplotlib.pyplot as plt

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