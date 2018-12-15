from feature_extractor import *

# Concrete Decorator
class CoordinatesFeature(FeatureDecorator):
    """
    Add responsibilities to the component.
    """
    def getFeatures(self):
        inner_df = self._component.getFeatures()
        print('Extracting Coordinates Features...')
        cur_df = self._dataset[['pu_lon', 'pu_lat', 'do_lon', 'do_lat']]
        return pd.concat([inner_df, cur_df], axis = 1)