from feature_extractor import *

# Concrete Decorator
class VectorDistanceFeature(FeatureDecorator):
    """
    Add responsibilities to the component.
    """
    def getFeatures(self):
        inner_df = self._component.getFeatures()
        print('Extracting VectorDistance Features...')
        cur_df = self._dataset['vec_dist']
        return pd.concat([inner_df, cur_df], axis=1)
