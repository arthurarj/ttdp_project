import pandas as pd

import geopy.distance
# Refer to: https://geopy.readthedocs.io/en/stable/#geopy.distance.geodesic

# Limiting coordinates for NYC
LAT_NORTH = 40.917577
LAT_SOUTH = 40.477399 
LON_EAST = -73.700272
LON_WEST = -74.259090 

# Limiting trip distance values
MIN_VEC_DIST = 0.01
MAX_RATIO = 25

# Uninteresting columns
columns_to_drop = ['VendorID', 
                   'passenger_count',
                   'RatecodeID', 
                   'store_and_fwd_flag', 
                   'payment_type', 
                   'extra', 
                   'mta_tax',
                   'tolls_amount',
                   'improvement_surcharge',
                   'PULocationID',
                   'DOLocationID']

# Let pandas know what features are dates
date_columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']

# A dict to rename features
columns_to_rename = { 'tpep_pickup_datetime':'pu_t',
                     'tpep_dropoff_datetime':'do_t',
                             'trip_distance':'trip_dist',
                          'pickup_longitude':'pu_lon',
                           'pickup_latitude':'pu_lat',
                         'dropoff_longitude':'do_lon',
                          'dropoff_latitude':'do_lat'}
# Numeric columns
numeric_columns = ['pu_lon', 'pu_lat', 'do_lon', 'do_lat']

def load_taxi_data_chunk(chunk = 10, raw = False):
    df = pd.DataFrame((pd.read_csv('../data/2016_Yellow_Taxi_Trip_Data.csv', chunksize=chunk, parse_dates=date_columns)
                     .get_chunk(chunk))
                     .drop(columns_to_drop, axis=1)
                     .rename(columns_to_rename,axis=1))
    print('Data loaded, now being cleaned and augmented.')

    # Clean data
    df = handle_missing_data(df)

    # Augment data
    df['duration'] = df.apply(lambda r: (r['do_t'] - r['pu_t']).seconds, axis=1)
    df['vec_dist'] = df.apply(lambda s : geopy.distance.geodesic((s.pu_lat, s.pu_lon),(s.do_lat, s.do_lon)).miles, axis=1)
    df['trip_ratio'] = df.trip_dist / df.vec_dist
    
    print('Data loaded with {0} entries and {1} columns'.format(df.index.size, df.columns.size))
    return df

def handle_missing_data(df):
    #df_size = df.index.size         # Track size of dataset 
    df.dropna(inplace=True)         # Remove rows with NaN values
    
    # Before performing operations, make sure everything is numeric
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    df.drop(df[df.pu_lat * df.pu_lon * df.do_lat * df.do_lat == 0].index, inplace=True)
    df.drop(df[(df.pu_lat == df.do_lat) | (df.pu_lon == df.do_lon)].index, inplace=True)
    df.drop(df[df.trip_dist == 0].index, inplace=True)
    #print('Size reduction from {0} to {1} ({2} samples dropped for missing data)'.format(df_size, df.index.size, df_size - df.index.size))
    
    return df

def handle_invalid_coordinates(df):
    # Latitudes must be in the [-90; 90] range
    df.drop(df[~df.pu_lat.between(-90,90)].index, inplace=True)
    df.drop(df[~df.do_lat.between(-90,90)].index, inplace=True)
    # Longitudes must be in the [-180; 180] range
    df.drop(df[~df.pu_lon.between(-180,180)].index, inplace=True)
    df.drop(df[~df.do_lon.between(-180,180)].index, inplace=True)

def handle_duration_outliers(df, threshold):
    df.drop(df[df.duration == 0].index, inplace=True)
    df.drop(df[df.duration > threshold].index, inplace=True)
    
def handle_spatial_outliers(df):
    df.drop(df[(df.pu_lat > LAT_NORTH) | (df.pu_lat < LAT_SOUTH)].index, inplace=True)
    df.drop(df[(df.pu_lon >  LON_EAST) | (df.pu_lon <  LON_WEST)].index, inplace=True)
    df.drop(df[(df.do_lat > LAT_NORTH) | (df.do_lat < LAT_SOUTH)].index, inplace=True)
    df.drop(df[(df.do_lon >  LON_EAST) | (df.do_lon <  LON_WEST)].index, inplace=True)

def handle_invalid_trips(df):
    df.drop(df[df.trip_ratio > MAX_RATIO].index, inplace=True)
    df.drop(df[df.vec_dist < MIN_VEC_DIST].index, inplace=True)