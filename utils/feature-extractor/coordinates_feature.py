from feature_extractor import *

# Square box around NYC
LAT_NORTH = 40.976897
LAT_SOUTH = 40.418079
LON_EAST = -73.700272
LON_WEST = -74.259090
assert LAT_NORTH-LAT_SOUTH == LON_EAST-LON_WEST
RANGE = LAT_NORTH-LAT_SOUTH

# Concrete Decorator
class CoordinatesFeature(FeatureDecorator):
    """
    Add responsibilities to the component.
    """
    def getFeatures(self):
        inner_df = self._component.getFeatures()
        print('\t Extracting Coordinates Features...')
        cur_df = self._dataset[['pu_lon', 'pu_lat', 'do_lon', 'do_lat']]

        # Min-Max Normalization
        self._dataset['pu_lon_n'] = (cur_df.pu_lon - LON_WEST)/RANGE
        self._dataset['pu_lat_n'] = (cur_df.pu_lat - LAT_SOUTH)/RANGE
        self._dataset['do_lon_n'] = (cur_df.do_lon - LON_WEST)/RANGE
        self._dataset['do_lat_n'] = (cur_df.do_lat - LAT_SOUTH)/RANGE

        return pd.concat([inner_df, cur_df], axis = 1)