"""Generate an animation of the cellular automaton Rule 30."""

import json
import os
import pathlib
import shutil
import subprocess
import tempfile

import colour
import cv2
import imageio
import numpy as np
import scipy.signal as sg
import tqdm

# Global parameters
CONFIG_PATH = 'config/full.json'
FFMPEG_PATH = '/usr/bin/ffmpeg'

# Load configuration
with open(CONFIG_PATH) as f:
    config = json.load(f)
    VIDEO_WIDTH = config['video_width']
    VIDEO_HEIGHT = config['video_height']
    SECS = config['secs']
    FPS = config['fps']
    PIXEL_SIZE = config['pixel_size']
    OUTPUT_PATH = config['output_path']
    # Trade-off speed and temp storage requirements
    COMP_LEVEL = config['comp_level']
    COLOURS = map(colour.Color, config['colours'])

# Constants
STATE_WIDTH = VIDEO_WIDTH // PIXEL_SIZE
STATE_HEIGHT = VIDEO_HEIGHT // PIXEL_SIZE
NUM_FRAMES = SECS * FPS


class Rule30:
    """A class for generating Rule 30."""

    neighbours = np.array([[1, 2, 4]], np.uint8)
    kernel = np.array([0, 1, 2, 3, 4, 0, 0, 0])
    colours = np.array([
        list(map(lambda x: round(255 * x), c.rgb)) for c in COLOURS
    ], np.uint8)

    def __init__(self, width, height):
        """Initialise the Rule 30 generator and set initial state.

        Args:
            width (int): State width
            height(int): State height
        """
        self.width = width
        self.height = height

        self.state = np.zeros((self.height, self.width), np.uint8)
        self.peak_height = 1
        self.state[-1, self.width // 2] = 2

        self.rgb = None
        self._update_rgb()

    def step(self):
        """Update the state and RGB representation."""
        self._update_state()
        self._update_rgb()
    
    def _update_state(self):
        """Update the state by applying Rule 30."""
        conv_row_alive = (self.state[-1, None, :] > 0).astype(np.uint8)
        rule_index = sg.convolve2d(conv_row_alive, self.neighbours,
                                   mode='same', boundary='wrap')
        new_row = self.kernel[rule_index]
        self.state = np.concatenate((self.state[1:], new_row))

        if self.peak_height < self.height:
            self.peak_height += 1
            self.state[-self.peak_height, self.width // 2] = 2

    def _update_rgb(self):
        """Convert the state to an RGB array."""
        self.rgb = self.colours[self.state]


class VideoConverter:
    """A class for converting frames of NumPy arrays to a video."""

    def __init__(self, fps=30):
        """Initialise the converter and create a temporary directory.

        Args:
            fps (int): Frames per second for the converted video
        """
        self.fps = fps
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.curr_frame = 0

    def add_frame(self, frame):
        """Adds a new frame to the video.

        Args:
            frame (uint8 NumPy array of shape: (video_height, video_width, 3))
                Data of the new frame as RGB. All frames must have the same
                dimensions.
        """
        frame_path = os.path.join(self.tmp_dir.name, f'{self.curr_frame}.png')
        imageio.imwrite(frame_path, frame, compress_level=COMP_LEVEL)
        self.curr_frame += 1

    def write(self, output_path):
        """Converts the accumulated frames to video and writes the result.

        Args:
            output_path: (string) Path where to save the video file
        """
        abs_tmp_dir_path = pathlib.Path(self.tmp_dir.name).absolute()
        abs_output_path = pathlib.Path(output_path).absolute()
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
        if OUTPUT_PATH[-4:] == '.mp4':
            subprocess.call([FFMPEG_PATH,
                            '-framerate', f'{self.fps}',
                            '-i', f'{abs_tmp_dir_path}/%d.png',
                            '-vcodec', 'libx264',
                            '-pix_fmt', 'yuv420p',
                            # Video quality, lower is better, but zero
                            # (lossless) doesn't work.
                            '-crf', '1',
                            '-y',  # overwrite output files without asking
                            abs_output_path
                            ])
        elif OUTPUT_PATH[-4:] == '.gif':
            subprocess.call([FFMPEG_PATH,
                            '-i', f'{abs_tmp_dir_path}/%d.png',
                            '-y',  # overwrite output files without asking
                            abs_output_path
                            ])
        else:
            raise NotImplementedError(
                "filetype not support"
            )
        self.tmp_dir.cleanup()
        print(f"Video written to: {abs_output_path}")


def main():
    converter = VideoConverter(fps=FPS)
    animation = Rule30(STATE_WIDTH, STATE_HEIGHT + 1)

    for __ in tqdm.trange(NUM_FRAMES // PIXEL_SIZE,
                          desc='Generating frames'):
        small_frame = animation.rgb
        enlarged_frame = cv2.resize(small_frame,
                                    (VIDEO_WIDTH, VIDEO_HEIGHT + PIXEL_SIZE),
                                    interpolation=cv2.INTER_NEAREST)
        for i in range(PIXEL_SIZE):
            converter.add_frame(enlarged_frame[i:(-PIXEL_SIZE + i)])
        animation.step()

    converter.write(OUTPUT_PATH)


if __name__ == '__main__':
    main()
