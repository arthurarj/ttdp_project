from feature_extractor import *

# Concrete Decorator
class NaiveTripDistanceFeature(FeatureDecorator):
    """
    Add responsibilities to the component.
    """
    def getFeatures(self):
        inner_df = self._component.getFeatures()
        print('Extracting NaiveTripDistance Features...')
        cur_df = self._dataset.apply(self._getNaiveTripDistanceFeature, axis = 1).rename('naive_trip_dist')
        self._dataset['naive_trip_dist'] = cur_df
        return pd.concat([inner_df, cur_df], axis = 1)

    def _getNaiveTripDistanceFeature(self, s):
        return s.vec_dist * 1.31 # Median estimator