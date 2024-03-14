# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:03:21 2023

@author: vincentkuo
"""

from pytube import YouTube
import os
  
url = "https://www.youtube.com/watch?v=ZT0lIsY0qkI"
target_path = "C:\\Users\\vincentkuo\\Downloads"

yt = YouTube(url)

video = yt.streams.filter(only_audio=True).first()

out_file = video.download(output_path=target_path)

base, ext = os.path.splitext(out_file)
new_file = base + '.mp3'
os.rename(out_file, new_file)

print("target path = " + (new_file))
print("mp3 has been successfully downloaded.")