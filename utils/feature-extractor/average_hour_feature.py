from feature_extractor import *

# Concrete Decorator
class AvgHourFeature(FeatureDecorator):
    """
    Add responsibilities to the component.
    """
    def getFeatures(self):
        inner_df = self._component.getFeatures()
        print('Extracting AvgHour Features...')
        
        cur_df = pd.DataFrame(index=self._dataset.index)
        cur_df['avg_hour'] = self._dataset.apply(self._getAvHourFeature, axis = 1)

        cur_df['avg_hour_sin'] = np.sin(cur_df.avg_hour * 2 * np.pi/ 24)
        cur_df['avg_hour_cos'] = np.cos(cur_df.avg_hour * 2 * np.pi/ 24)

        self._dataset['avg_hour'] = cur_df.avg_hour
        return pd.concat([inner_df, cur_df], axis = 1)

    def _getAvHourFeature(self, s):
    	return pd.Timestamp(np.mean((s.pu_t.value, s.do_t.value))).hour