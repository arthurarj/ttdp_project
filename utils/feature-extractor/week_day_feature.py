from feature_extractor import *

# Concrete Decorator
class WeekDayFeature(FeatureDecorator):
    """
    Add responsibilities to the component.
    """
    def getFeatures(self):
        inner_df = self._component.getFeatures()
        print('Extracting WeekDay Features...')

        cur_df = pd.DataFrame(index=self._dataset.index)
        cur_df['week_day'] = self._dataset.apply(self._getWeekDayFeature, axis = 1)
        
        cur_df['week_day_sin'] = np.sin(cur_df.week_day * 2 * np.pi/ 7)
        cur_df['week_day_cos'] = np.cos(cur_df.week_day * 2 * np.pi/ 7)

        self._dataset['week_day'] = cur_df.week_day
        return pd.concat([inner_df, cur_df], axis = 1)

    def _getWeekDayFeature(self, s):
    	return pd.Timestamp(np.mean((s.pu_t.value, s.do_t.value))).weekday()