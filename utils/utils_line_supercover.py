import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class GridMap():
    
    class LatLonScaler():
        def fit(self, size, lat_north, lat_south, lon_east, lon_west):
            self.p_max = np.array((lon_east, lat_north))
            self.p_min = np.array((lon_west, lat_south))
            self.range = self.p_max - self.p_min
            self.scale = size
        def transform(self, point):
            return self.scale * (point - self.p_min)/(self.p_max - self.p_min)
        def inverse_transform(self, point):
            return (self.p_max - self.p_min) * point / self.scale + self.p_min
    
    def __init__(self, size, bbox):
        self.hsize = size[0]
        self.vsize = size[1]
        self.scaler = self.LatLonScaler()
        self.scaler.fit(size, *bbox)
        
    def plot_path(self, p1, p2, need_norm = True, cells = None):
        if need_norm == True:
            p1 = self.scaler.transform(p1)
            p2 = self.scaler.transform(p2)
        
        fig = plt.figure(figsize=(4, 4 * self.vsize / self.hsize))
        ax = fig.add_subplot(1,1,1)

        if cells is not None:
            ax = fig.gca()
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
        
    def plot_m_heat(self, m_map):
        fig = plt.figure(figsize=(5, 5 * self.vsize / self.hsize))
        ax = fig.add_subplot(1, 1, 1)

        r,c = m_map.shape
        for i in range(r):
            for j in range(c):
                ifm_map[i,j]
                
        print(median_map.shape)
        #         if cells is not None:
#             ax = fig.gca()
#             for cell in cells:
#                 ax.add_patch(Rectangle((cell[0],cell[1]),1,1, facecolor='k', edgecolor='k',alpha=0.25))
#        
#         plt.scatter(p1[0], p1[1],c='r')
#         plt.scatter(p2[0], p2[1],c='g')
#         plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='k')
        
        ax.set_xticks(range(0, self.hsize + 1))
        ax.set_yticks(range(0, self.vsize + 1))
        plt.xlim(0, self.hsize)
        plt.ylim(0, self.vsize)
        plt.grid()
        plt.show()    
        
    def getSuperCover(self, p1, p2, need_norm=True):
        # Implemented based on: http://playtechs.blogspot.com/2007/03/raytracing-on-grid.html
        
        if need_norm == True:
            p1 = self.scaler.transform(p1)
            p2 = self.scaler.transform(p2)
        
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
    
    def _mapDataSampleToGrid(self, sample, assigning_map):
        pu = np.array([sample.pu_lon, sample.pu_lat])
        do = np.array([sample.do_lon, sample.do_lat])
        covered_cells = self.getSuperCover(pu,do)
        for cell in covered_cells:
            key = (cell[0],cell[1])
            if key in assigning_map:
                assigning_map[(cell[0],cell[1])].append(sample.name)
            else:
                assigning_map[(cell[0],cell[1])] = [sample.name]
        return len(covered_cells)

    def mapDataToGrid(self, dataset):
        # This method assigns each sample to the grid cells for which that trip passes through
        assigning_map = {}
        sample_path_length = dataset.apply(lambda r : self._mapDataSampleToGrid(r, assigning_map), axis=1)
        return assigning_map, sample_path_length
    
    def apply(self, dataset, assigning_map, func):
        output = np.nan * np.empty((self.hsize, self.vsize))
        for (i,j), value in assigning_map.items():
            output[i][j] = getattr(dataset.loc[assigning_map[(i,j)]].trip_ratio, func)()
        return output