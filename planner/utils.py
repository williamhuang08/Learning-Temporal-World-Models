""" Functions to Save GIFs of the Planned Trajectory (functions from prev utils in skill learning paper)"""

from PIL import Image
import matplotlib.pyplot as plt
import torch
import imageio

def make_gif(frames, name):
    frames = [Image.fromarray(image) for image in frames]
    frame_one = frames[0]
    frame_one.save(name+'.gif', format="GIF", append_images=frames,save_all=True, duration=100)

def make_video(frames,name):
	writer = imageio.get_writer(name+'.mp4', fps=20)

	for im in frames:
		writer.append_data(im)
	writer.close()

