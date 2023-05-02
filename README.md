Efficient near realtime animation of soundwave and simulatenous 
playback of audio, primarily for use in Jupyter notebooks. 
Approximately 10x faster than naive matplotlib animation approaches.
Guarantees no out of sync audio by quickly rendering as an MPEG4 video.
Time to render 30 FPS, 0.25 second viewport over a 7 second clip is 
approximately 2-3 seconds.

Now allows multiple tracks to be animated in the same video, and 
added a Jupyter Widget controller for longer audio track seeking.

```
#pip install --upgrade --no-deps --force-reinstall --quiet 'git+https://github.com/davies-w/pyavanimate.git'

import pyavanimate

stereo_song_amp_wav_1, stereo_song_amp_aac_1, song_rate, song_duration = pyavanimate.make_test_song(pow(0.5,4), 0.75)
stereo_song_amp_wav_2, stereo_song_amp_aac_2, song_rate, song_duration = pyavanimate.make_test_song(1.0, 0.25)
tracks = [stereo_song_amp_aac_1, stereo_song_amp_aac_2]
pyavanimate.make_ipywidget_player(tracks, song_rate, song_duration)
```
