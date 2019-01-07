from feature_extractor import *

# Concrete Decorator
class AvgSpeedFeature(FeatureDecorator):
    """
    Add responsibilities to the component.
    """
    def getFeatures(self):
        inner_df = self._component.getFeatures()
        print('Extracting AvgSpeed Features...')

        cur_df = pd.DataFrame(index=self._dataset.index)
        cur_df['avg_speed_mph'] = self._dataset.apply(self._getAvSpeedFeature, axis = 1)

        self._dataset['avg_speed_mph'] = cur_df.avg_speed_mph
        return pd.concat([inner_df, cur_df], axis = 1)

    def _getAvHourFeature(self, s):
    	return  3600 * dataset.trip_dist / dataset.duration