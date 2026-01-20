# Source - https://stackoverflow.com/a
# Posted by Masoud, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-02, License - CC BY-SA 4.0

import cv2
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import sys

from PIL import Image, ImageSequence
import os
from pathlib import Path

def horiz_swap(img):
    hs_img = np.roll(img, img.shape[1]//2, axis = 1)
    
    return hs_img

def load_frames(img: Image, mode='RGBA'):
    return np.array([
        np.array(frame.convert(mode))
        for frame in ImageSequence.Iterator(img)
    ])


if __name__=='__main__':    
    # Get directory name from command line argument
    if len(sys.argv) < 2:
        print("Usage: python horiz_swap.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    ext = sys.argv[2]
    pattern = sys.argv[3] + ext
    
    # Process all .gif files in the directory
    print(directory)
    path = Path(directory)
    for im_file in path.glob(pattern):
        if not "horizswap" in im_file.stem:
            print(f"Processing {im_file.name}...")
            with Image.open(im_file) as im:
                frames = load_frames(im)
                hs_frames = []
                for img in frames:
                    hs_frames.append(horiz_swap(img))
                
                # Save the horizswapped images to file:
                hs_frames = [Image.fromarray(frame) for frame in hs_frames]
                hs_dir = Path(im_file.parent)
                hs_dir.mkdir(exist_ok=True)
                hs_frames[0].save(hs_dir / (im_file.stem + '_horizswap' + ext),
                    save_all = True, append_images = hs_frames[1:],
                    optimize = False, duration = 10,
                )
