# pyavanimate/pyavanimate/pyavanimate.py
#
# Copyright 2023 Winton Davies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.video.io import html_tools
from IPython.display import Audio, Video, display, Image
import PIL.Image
from io import BytesIO
import math
import time

def showarray(a, fmt='png'):
    f = BytesIO()
    pil = PIL.Image.fromarray(a)
    pil.save(f, fmt)
    display(Image(data=f.getvalue()))

def generate_intervals(N, M, interval_size, clip_min=False, clip_max=False):
    start = math.ceil(N / interval_size) * interval_size
    end = math.floor(M / interval_size) * interval_size
    if clip_min and start == N:
     start += interval_size
    if clip_max and end == M:
     end -= interval_size
    return [round(x, 2) for x in np.arange(start, end+interval_size, interval_size)]

def note(frequency, length, amplitude=1, sample_rate=22050):
    time_points = np.linspace(0, length, length*sample_rate)
    data = np.sin(2*np.pi*frequency*time_points)
    data = amplitude*data
    return data

def make_single_image(current_time, stereo_song_amps, song_rate, time_window, width, height, song_duration = None, intervals=0.05):
    fig = plt.figure(figsize = (width, height), layout="constrained")
    ax = fig.add_subplot(111) #so we can see the x-axis against HTML5 player.
    ax.set_xlabel(" ", fontsize=36)      # make it stand up against player.
    ax.set_ylim(-1, 1)

    x, y = [], []
    if song_duration:                            # we're making one long image.
      # this is how long the x-axis items are.
      #print("making one big image!")
      min_x = 0
      max_x = song_duration
      min_samples_x = 0
      max_samples_x = int(song_duration*song_rate)
      x = np.arange(min_samples_x, max_samples_x, 1)/song_rate   # x-array as a bunch of fractions of seconds.
      ax.set_xlim(min_x, max_x+1*time_window)                      # where the left and right y-axis intercept
      ax.set_xticks(generate_intervals(min_x, max_x, intervals, clip_min=True)) # tick range
      y = stereo_song_amps[min_samples_x:max_samples_x]          # y array
    else:
      min_x = current_time
      max_x = current_time+time_window
      min_samples_x = int(current_time*song_rate)
      max_samples_x = int((current_time+time_window)*song_rate)
      x = np.arange(min_samples_x, max_samples_x, 1)/song_rate    # x-array as a bunch of fractions of seconds.
      ax.set_xlim(min_x, max_x)                                   # where the left and right y-axis intercept
      ax.set_xticks(generate_intervals(min_x, max_x, intervals))  # tick range
      y = stereo_song_amps[min_samples_x:max_samples_x]           # y array

    if (x.shape[0] != y.shape[0]):                                # make sure x and y are same length
       x = np.resize(x, (y.shape[0],)) 

    ax.plot(x,y)
    plt.close()
    return mplfig_to_npimage(fig)

def display_animation_slow(stereo_song_amps, song_duration, song_rate, fps, time_window, width, height):
  # in theory, song_duration = len(stereo_song_amps)/song_rate
  total_num_frames = song_duration*fps
  frames = []
  for i in range(int(total_num_frames)):
    frames.append(make_single_image(i/fps,
                                    stereo_song_amps, 
                                    song_rate, 
                                    time_window, 
                                    width, 
                                    height))
  # *Have to* have FPS **and** Duration Array both set.
  animation = ImageSequenceClip(frames, fps=fps, durations=[1.0/fps]*total_num_frames)
  animation.audio = AudioArrayClip(stereo_song_amps, song_rate)

  display(animation.ipython_display(rd_kwargs=dict(logger=None, preset="ultrafast", fps=fps, audio_fps=song_rate)))

  
def display_animation_fast(song_amp_aac, song_rate, view_window_secs=0.25, view_port_inches=10, height=2):
  
  song_duration = int(len(song_amp_aac)/song_rate)
  plot_width = int(view_port_inches * song_duration/view_window_secs)
  #print (f"width={plot_width}")
  #start_time = 0
  fps=30

  total_num_frames = song_duration*fps
  start_time = time.time()
  im = make_single_image(start_time, song_amp_aac, song_rate, view_window_secs , width = plot_width, height = height, song_duration=song_duration)
  #print(f"matplotlib {(time.time() - start_time) } seconds.")
  mid_y = 105 # Note this is y indexed such that 0 is top of image, max is bot.
            # thus 0 is close to the +1.0 image, and 120 or so is close to -1.0
  max_x =-1
  x_search_limit = 100
  for x in range(0, x_search_limit, 1):
    if np.array_equal(im[mid_y, x, :], np.array([0, 0, 0])):
      front_x = x + 1

  for x in range(max_x, (max_x - x_search_limit), -1):
    if np.array_equal(im[mid_y, x, :], np.array([0, 0, 0])):
      back_x = x - 1

  front = im[:, 0:front_x, :]
  #print(f"front_x {front_x},{front.shape[1]}")
  back  = im[:, back_x:-1, :]
  #print(f"back_x {back_x}, {back.shape[1]}")

  #
  #    We just need to move left 1/30th of a second each time, keeping the 0.25 
  #    potentially, we could even just keep adding and removing that 1/30th of a
  #    frame.
  # 
  pixels_per_second = int(np.round((im.shape[1] - (front_x + abs(back_x))) / (song_duration + view_window_secs)))
  #print(f"pixels per second of song: {pixels_per_second}")
  pixels_per_viewport = int(np.round(pixels_per_second*view_window_secs))
  #print(f"pixels per view port: {pixels_per_viewport}")
  pixels_shift_left_per_frame = int(np.round(pixels_per_second/fps))
  #print(f"pixels to shift left per frame: {pixels_shift_left_per_frame}")
  expected_width = pixels_per_viewport+front_x-back_x-1

  frames = []
  for frame_number in range(total_num_frames+1):
    start_pos = front_x+frame_number*pixels_shift_left_per_frame
    end_pos =   start_pos+pixels_per_viewport
    middle = im[:, start_pos:end_pos, :]
    frame = np.concatenate([front, middle, back], axis=1)
    # print(f"{frame_number} {frame.shape[1]} expected: {expected_width}")
    if frame.shape[1] != expected_width:
      middle = im[:,back_x-pixels_per_viewport+1:back_x+1,:]
      frame = np.concatenate([front, middle, back], axis=1)   
    #showarray(frame)
    frames.append(frame)
  
  #print(f"framemaking {(time.time() - start_time) } seconds.")
  animation = ImageSequenceClip(frames, fps=fps, durations=[1.0/fps]*len(frames))
  #print(f"animation making {(time.time() - start_time) } seconds.")
  animation.audio = AudioArrayClip(song_amp_aac, song_rate)
  # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo
  movie=animation.ipython_display(rd_kwargs=dict(logger=None, preset="veryfast", fps=fps, audio_fps=song_rate))
  #print(f"rendering {(time.time() - start_time) } seconds.")
  display(movie)
  
 
  
def make_test_note(frequency, length, amplitude=1, sample_rate=22050):
    time_points = np.linspace(0, length, length*sample_rate)
    data = np.sin(2*np.pi*frequency*time_points)
    data = amplitude*data
    return data

def make_test_song():
  song_rate = 44100
  song_amp =  np.concatenate(
             (make_test_note(440.00, 1, 0.9, song_rate),  #A 
              make_test_note(493, 1, 0.8, song_rate),     #B .88
              make_test_note(523, 1, 0.7, song_rate),     #C .25
              make_test_note(587, 1, 0.6, song_rate),     #D .33
              make_test_note(659, 1, 0.5, song_rate),     #E .25
              make_test_note(698, 1, 0.4, song_rate),     #F .46
              make_test_note(783, 1, 0.3, song_rate),     #G .99
             ),             
             axis=-1)
             
  song_duration = 7
  song_amp_aac = song_amp.reshape((song_amp.shape[0],1)) 
  #noise = np.random.normal(0, 0.005, song_amp_aac.shape)
  #song_amp_aac = song_amp_aac + noise
  song_amp_aac = np.concatenate([song_amp_aac, song_amp_aac],1)

  song_amp_wav = song_amp * 32767
  song_amp_wav = song_amp_wav.clip(-32766,32767)
  song_amp_wav = song_amp_wav.astype('int16')
  return song_amp_wav, song_amp_aac, song_rate, song_duration
  
def test_slow_song():
  song_amp_wav, song_amp_aac, song_rate, song_duration = make_test_song()
  display_animation_slow(song_amp_aac, song_duration=song_duration, song_rate=song_rate, fps=30, time_window=0.25, width=10, height=2)
  
def test_fast_song():
  song_amp_wav, song_amp_aac, song_rate, song_duration = make_test_song()
  display_animation_fast(song_amp_aac, song_rate)

  
