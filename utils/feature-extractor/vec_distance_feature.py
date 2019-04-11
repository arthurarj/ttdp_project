from feature_extractor import *

# Square box around NYC
LAT_NORTH = 40.976897
LAT_SOUTH = 40.418079
LON_EAST = -73.700272
LON_WEST = -74.259090
assert LAT_NORTH-LAT_SOUTH == LON_EAST-LON_WEST
RANGE = LAT_NORTH-LAT_SOUTH

# Concrete Decorator
class VectorDistanceFeature(FeatureDecorator):
    """
    Add responsibilities to the component.
    """
    def getFeatures(self):
        inner_df = self._component.getFeatures()
        print('\t Extracting VectorDistance Features...')
        cur_df = self._dataset['vec_dist']
        return pd.concat([inner_df, cur_df], axis=1)
