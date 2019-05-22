#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv
from PIL import Image
from io import BytesIO
import pandas as pd
from urllib.request import urlopen
import tqdm

def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  key_url_list = [line[:2] for line in csvreader]
  return key_url_list[1:]  # Chop off header


def DownloadImage(key_url):
  out_dir = 'images-validation'
  (url, key) = key_url
  filename = os.path.join(out_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    #print('Image %s already exists. Skipping download.' % filename)
    return 0

  try:
    response = urlopen(url)
    image_data = response.read()
  except:
    #print('Warning: Could not download image %s from %s' % (key, url))
    return 1

  try:
    pil_image = Image.open(BytesIO(image_data))
  except:
    #print('Warning: Failed to parse image %s' % key)
    return 1

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except:
    #print('Warning: Failed to convert image %s to RGB' % key)
    return 1

  try:
    pil_image_rgb.save(filename, format='JPEG', quality=90)
  except:
    #print('Warning: Failed to save image %s' % filename)
    return 1
  #print(key)
  return 0

def Run():
  train = pd.read_csv('features-validation.csv')
  url = train['image_thumbnail']
  key = train['id']
  key_url = pd.concat([url, key], axis=1)
  key_url_list = key_url.values

  pool = multiprocessing.Pool(processes=100)
  failures = sum(tqdm.tqdm(pool.imap_unordered(DownloadImage, key_url_list), total=len(key_url_list)))
  print('Total number of download failures:', failures)
  pool.close()
  pool.terminate()

if __name__ == '__main__':
  Run()
