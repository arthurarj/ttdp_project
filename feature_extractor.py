# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:07:40 2019

@author: Arthur
"""
import pickle as pkl
import pandas as pd
import numpy as np

import argparse
import logging

import glob
import datetime

import os
import sys
sys.path.insert(0, 'utils/feature-extractor')

from feat_extraction import *
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='Feature extractor.')
parser.add_argument('input_dir', 
                    metavar='s', type=str, nargs=1,
                    help='The path for the data CSV file.')
parser.add_argument('output_dir', 
                    metavar='o', type=str, nargs=1,
                    help='Output directory.')

# Fetching arguments
args = parser.parse_args()
input_path = args.input_dir[0]
output_dir = args.output_dir[0]

# log_file_name = output_dir + '/feat_extractor_dump_{}.log'.format(''.join(c for c in str(datetime.datetime.now()) if c.isdigit())[:14])
# logging.basicConfig(filename=log_file_name,
#                     format='%(asctime)s | %(levelname)s | %(message)s',
#                     datefmt='%m/%d/%Y %I:%M:%S %p',
#                     level=logging.INFO)

# Get all .p files
p_files = glob.glob(input_path + "/*.p")

i = 0
for p_file_path in p_files:
	input_file_name = os.path.split(p_file_path)[1]
	output_file_name = 'processed_' + input_file_name

	if os.path.isfile(output_dir + output_file_name):
		print("File {0} was already processed, so skip it.".format(input_file_name))
		continue

	print("Processing file {0}".format(input_file_name))


	dataset = pkl.load(open(p_file_path, 'rb'))
	
	# Instantiate a (concrete) component
	featExtractor = FeatureExtractor(dataset=dataset)

	# Decorate with features
	featExtractor = CoordinatesFeature(featExtractor) 		#Normalized
	featExtractor = VectorDistanceFeature(featExtractor)	#Normalized
	featExtractor = AvgHourFeature(featExtractor)
	featExtractor = WeekDayFeature(featExtractor)

	# Extract them
	features = featExtractor.getFeatures()

	# Dump it
	pkl.dump(dataset, open(output_dir + output_file_name, 'wb'))

	i = i + 1
#	if i < 30:
#		break