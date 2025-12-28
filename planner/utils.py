""" Functions to Save GIFs of the Planned Trajectory"""

from PIL import Image
import cv2

def make_gif(frames, name):
    frames = [Image.fromarray(image) for image in frames]
    frame_one = frames[0]
    frame_one.save(name+'.gif', format="GIF", append_images=frames,save_all=True, duration=100)

    