import matplotlib.pyplot as plt
import matplotlib.patches as patches
import folium.plugins as plugins
import folium
from numpy.random import randint

# Limiting coordinates for NYC
LAT_NORTH = 40.917577
LAT_SOUTH = 40.477399
LON_EAST = -73.700272
LON_WEST = -74.259090

# Set a maximum number of points to be rendered by Folium in Jupyter
MAX_FOL = 25000

# Assume that data has been cleaned 

def plot_coordinates_auto(df, segments=False, focus=False):
    fig = plt.figure(figsize=(10,10))
    plt.scatter(df['do_lon'], df['do_lat'], color='g', s=2, alpha = 1)
    plt.scatter(df['pu_lon'], df['pu_lat'], color='r', s=2, alpha = 0.5)
    if segments:
        plt.plot((df['pu_lon'],df['do_lon']),(df['pu_lat'],df['do_lat']), color='k', alpha=.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def plot_coordinates(df, segments=False, focus=False, bbox = False):
    fig = plt.figure(figsize=(10,10))
    plt.scatter(df['do_lon'], df['do_lat'], color='g', s=2, alpha = 1)
    plt.scatter(df['pu_lon'], df['pu_lat'], color='r', s=2, alpha = 0.5)
    if segments:
        plt.plot((df['pu_lon'],df['do_lon']),(df['pu_lat'],df['do_lat']), color='k', alpha=.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if focus:
        plt.xlim((-74.05, -73.9))
        plt.ylim((40.7, 40.85))
    else:
        plt.xlim((LON_WEST - 0.01, LON_EAST + 0.01))
        plt.ylim((LAT_SOUTH - 0.01, LAT_NORTH + 0.01))
        
    if bbox:
        ax = fig.gca()
        ax.add_patch(patches.Rectangle((LON_WEST, LAT_SOUTH), LON_EAST - LON_WEST, LAT_NORTH - LAT_SOUTH, 
                                       linewidth=1, edgecolor='r', facecolor='k', alpha=0.25))
    plt.show()
    
def folium_map(data):
    d = data
    if len(data) > MAX_FOL:
        sample = randint(0, len(data), MAX_FOL)
        d = data[sample] 
    heatmap = plugins.HeatMap(d)
    taxi_map = folium.Map(location=[40.763940, -73.88], tiles='stamentoner', zoom_start=10, max_zoom=20)
    heatmap.add_to(taxi_map)
    return taxi_map