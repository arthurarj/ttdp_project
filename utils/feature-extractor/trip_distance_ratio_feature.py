from feature_extractor import *

# Concrete Decorator
class TripDistanceRatioFeature(FeatureDecorator):
    """
    Add responsibilities to the component.
    """
    def getFeatures(self):
        inner_df = self._component.getFeatures()
        print('Extracting TripDistanceRatio Features...')
        cur_df = self._dataset['trip_ratio']
        return pd.concat([inner_df, cur_df], axis=1)
