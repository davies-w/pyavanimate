# pyavanimate/setup.py
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

from setuptools import setup
desc = \
'''
Efficient near realtime animation of soundwave and simulatenous 
playback of audio, primarily for use in Jupyter notebooks. 
Approximately 10x faster than naive matplotlib animation approaches.
Guarantees no out of sync audio by quickly rendering as an MPEG4 video.
Time to render 30 FPS, 0.25 second viewport over a 7 second clip is 
approximately 2-3 seconds.
'''

setup(
        name='pyavanimate',
        version='0.0.1',
        description=desc,
        url='git@github.com:davies-w/pyavanimate.git',
        author='Winton Davies',
        author_email='wdavies@cs.stanford.edu',
        license='Apache License 2.0',
        install_requires=["moviepy", "matplotlib", "numpy", "IPython", "PIL", "ipywidgets", "io", "math", "traceback", "functools", "time"],
        packages=['pyavanimate'],
        zip_safe=True
    )
