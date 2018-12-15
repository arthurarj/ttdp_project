import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from timeit import default_timer as timer

# Limiting coordinates for NYC
LAT_NORTH = 40.917577
LAT_SOUTH = 40.477399 
LON_EAST = -73.700272
LON_WEST = -74.259090 

class GridMap():

    # Class intended for scaling
    class LatLonScaler():
        def fit(self, size, lat_north, lat_south, lon_east, lon_west):
            self.p_max = np.array((lon_east, lat_north))
            self.p_min = np.array((lon_west, lat_south))
            self.range = self.p_max - self.p_min
            self.scale = size
            self.p_min_extended = np.array((lon_west, lat_south, lon_west, lat_south))
            self.p_max_extended = np.array((lon_east, lat_north, lon_east, lat_north))
            self.scale_extended = (*size, *size)
        def transform(self, point):
            return self.scale * (point - self.p_min)/(self.p_max - self.p_min)
        def transform_pts(self, points):
            return self.scale_extended * (points - self.p_min_extended)/(self.p_max_extended - self.p_min_extended)
                
        def inverse_transform(self, point):
            return (self.p_max - self.p_min) * point / self.scale + self.p_min
    
    def __init__(self, size, bbox):
        self.hsize = size[0]
        self.vsize = size[1]
        self.scaler = self.LatLonScaler()
        self.scaler.fit(size, *bbox)
        self.grid_model = None
        
### Private methods
    
    def _getSuperCover(self, p1, p2, need_norm=True):
        # Implemented based on: http://playtechs.blogspot.com/2007/03/raytracing-on-grid.html
        
        # Absolute differnce
        d = np.abs(p2 - p1)
        dt = 1/d
        
        # Starter cell
        p = np.floor(p1).astype('int')
        
        # Parametrized line
        t = 0
        
        # The line passes through n cells
        n = 1
        
        # Horizontal axis
        if d[0] == 0:
            # Displacement was solely vertical
            x_step = 0
            d_next_h = dt[0]
        elif p2[0] > p1[0]:
            # Eastbound
            x_step = +1
            n = n + (np.floor(p2[0]) - np.floor(p1[0]))
            d_next_h = (np.ceil(p1[0]) - p1[0]) * dt[0]
        else:
            # Westbound
            x_step = -1
            n = n - (np.floor(p2[0]) - np.floor(p1[0]))
            d_next_h = (p1[0] - np.floor(p1[0])) * dt[0]
        
        # Vertical axis
        if d[1] == 0:
            # Displacement was solely horizontal
            y_step = 0
            d_next_v = dt[1]
        elif p2[1] > p1[1]:
            # Northbound
            y_step = +1
            n = n + (np.floor(p2[1]) - np.floor(p1[1]))
            d_next_v = (np.ceil(p1[1]) - p1[1]) * dt[1]
        else:
            # Southbound
            y_step = -1
            n = n - (np.floor(p2[1]) - np.floor(p1[1]))
            d_next_v = (p1[1] - np.floor(p[1])) * dt[1]
              
        cells = []
        for i in range(int(n),0,-1):
            cells.append(np.copy(p))
            if d_next_v < d_next_h:
                # Vertical cell is closer
                p[1] = p[1] + y_step
                d_next_v = d_next_v + dt[1]
            else:
                # Horizontal cell is closer
                p[0] = p[0] + x_step
                d_next_h = d_next_h + dt[0]
        
        return np.array(cells)
    
    def _mapDataSampleToGrid(self, index, sample, assigning_map, need_norm=False):
        covered_cells = self._getSuperCover(sample[[0,1]], sample[[2,3]])
        for cell in covered_cells:
            key = (cell[0],cell[1])
            if key in assigning_map:
                assigning_map[(cell[0],cell[1])].append(index)
            else:
                assigning_map[(cell[0],cell[1])] = [index]
        return len(covered_cells)
    
### Public methods
    def plot_path(self, p1, p2, need_norm = True, show_cells = True):
        if need_norm == True:
            p1 = self.scaler.transform(p1)
            p2 = self.scaler.transform(p2)
        
        fig = plt.figure(figsize=(4, 4 * self.vsize / self.hsize))
        ax = fig.add_subplot(1,1,1)

        if show_cells is True:
            ax = fig.gca()
            cells = self._getSuperCover(p1, p2)
            for cell in cells:
                ax.add_patch(Rectangle((cell[0],cell[1]),1,1, facecolor='k', edgecolor='k',alpha=0.25))
        
        plt.scatter(p1[0], p1[1],c='r')
        plt.scatter(p2[0], p2[1],c='g')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='k')
        
        ax.set_xticks(range(0, self.hsize + 1))
        ax.set_yticks(range(0, self.vsize + 1))
        ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)    

        plt.xlim(0, self.hsize)
        plt.ylim(0, self.vsize)
        plt.grid()
        plt.show()

    def normalizeData(self, dataset):
        print("Normalizing data...")
        # Data should be a numpy array organized as pu_lon	pu_lat	do_lon	do_lat
        normalized_data = self.scaler.transform_pts(dataset)
        print("Finished.")
        return normalized_data
            
    def fitGrid(self, dataset, func, need_norm = True):
        # This method performs two steps:
        #  1) Assigns each sample to the grid cell(s) through which that trip passes
        #  2) For each cell, compute func through all samples assigned to that cell
        print("Started fitting grid of size {0}x{1}".format(self.hsize, self.vsize))
        
        if need_norm == True:
            print("Normalizing data...")
            norm_dataset = dataset[:,[0,1,2,3]].copy()
            norm_dataset = self.scaler.transform_pts(norm_dataset)

        print("Building assigning map...")
        assigning_map = {}
        start = timer()
        [self._mapDataSampleToGrid(i, row, assigning_map) for i, row in enumerate(norm_dataset)]
        print("Time spent:", timer() - start) 
            
        print("Reducing map...")
        prior = getattr(np, func)(dataset[:,-1])
        self.grid_model = prior * np.ones((self.hsize, self.vsize))
        for (i,j), value in assigning_map.items():
            self.grid_model[i][j] = getattr(np, func)(dataset[:,-1][assigning_map[(i,j)]])
        print("Finished.\n")
        return self.grid_model

    def _predictFactor(self, sample):
        cells = self._getSuperCover(sample[[0,1]], sample[[2,3]])
        return np.mean([self.grid_model[i][j] for (i,j) in cells])
    
    def predictFactor(self, dataset, need_norm = True):
        print("Started prediction on grid of size {0}x{1}".format(self.hsize, self.vsize))

        if need_norm == True:
            print("Normalizing data...")
            norm_dataset = dataset[:,[0,1,2,3]].copy()
            norm_dataset = self.scaler.transform_pts(norm_dataset)

        print("Prediction has started.")
        pred = np.array([self._predictFactor(row) for _, row in enumerate(norm_dataset)])
        print("Finished.\n")
        return pred