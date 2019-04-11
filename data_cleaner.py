# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:06:53 2019

@author: Arthur
"""
from utils.utils_data_cleaning import *

import argparse
import logging
import pickle as pkl

import datetime

parser = argparse.ArgumentParser(description='Process CSV file into chunks.')
parser.add_argument('data_path', 
                    metavar='s', type=str, nargs=1,
                    help='The path for the data CSV file.')
parser.add_argument('output_dir', 
                    metavar='o', type=str, nargs=1,
                    help='Output directory.')
# Fetching arguments
args = parser.parse_args()
data_path = args.data_path[0]
output_dir = args.output_dir[0]

log_file_name = output_dir + 'data_dump_{}.log'.format(''.join(c for c in str(datetime.datetime.now()) if c.isdigit())[:14])
logging.basicConfig(filename=log_file_name,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

total_trips_read = 0
total_trips_saved = 0

chunk_size = 100000

chunks_to_skip = 899 #260
rows_to_skip = chunks_to_skip * chunk_size

start = chunks_to_skip
#i = 0
i = start

for data_chunk in pd.read_csv(data_path, 
                              chunksize=chunk_size,
                              skiprows=range(1, rows_to_skip + 1),
                              parse_dates=date_columns):

    print('#{}: '.format(i), end='')
    
    total_trips_read = total_trips_read + len(data_chunk)
    size_drop = [len(data_chunk)]

    data_chunk.drop(columns_to_drop, axis=1, inplace=True)
    data_chunk.rename(columns_to_rename,axis=1, inplace=True)

    # Handle Missing Data
    prior_size = data_chunk.shape[0]

    data_chunk = handle_missing_data(data_chunk)

    if len(data_chunk) == 0:
        size_drop.append(len(data_chunk))
        logging.info("Chunk {0}: Size went {1} ".format(i, size_drop))
        print('All samples have missing data.')
        i = i + 1
        continue

    # Handle invalid coordinates
    handle_invalid_coordinates(data_chunk)

    # New columns
    data_chunk['duration'] = data_chunk.apply(lambda r: (r['do_t'] - r['pu_t']).seconds, axis=1)
    data_chunk['vec_dist'] = data_chunk.apply(lambda s : geopy.distance.geodesic((s.pu_lat, s.pu_lon),(s.do_lat, s.do_lon)).miles, axis=1)
    data_chunk['trip_ratio'] = data_chunk.trip_dist / data_chunk.vec_dist

    # Handle outliers and invalid trips
    handle_duration_outliers(data_chunk, 7200)
    size_drop.append(len(data_chunk))
    handle_spatial_outliers(data_chunk)
    size_drop.append(len(data_chunk))
    handle_invalid_trips(data_chunk)
    size_drop.append(len(data_chunk))

    # Reset index
    data_chunk.reset_index(inplace=True)
    data_chunk.rename({'index':'original_index'},axis=1, inplace=True)

    print(size_drop, end='')
    
    # Dump it
    pkl.dump(data_chunk, open(output_dir + 'ttd_chunk_{0}.p'.format(i), 'wb'))
    total_trips_saved = total_trips_saved + len(data_chunk)

    print(" ({} dropped)".format(prior_size - data_chunk.shape[0]), end='\n')

    logging.info("Chunk {0}: Size went {1}".format(i, size_drop))

    i = i + 1
    #if i > 260:
    #    break

logging.info("Summary: {0} trips read, {1} saved.".format(total_trips_read, total_trips_saved))
print("Summary: {0} trips read, {1} saved.".format(total_trips_read, total_trips_saved))
print("Done.")