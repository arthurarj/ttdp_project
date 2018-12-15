from feature_extractor import *
from utils.utils_line_supercover import *

import pickle as pkl

# Concrete Decorator
class GridTripDistanceFeature(FeatureDecorator):
	"""
	Add responsibilities to the component.
	"""
	def __init__(self, dataset):
		super(GridTripDistanceFeature, self).__init__(dataset)
		self.grid_map = None

	def getFeatures(self):
		inner_df = self._component.getFeatures()
		print('Extracting GridTripDistance Features...')

		self.grid_map = pkl.load(open('../distance_models/grid_trip_distance_model_{0}.mdl'.format(300), 'rb'))
		ratios = self.grid_map.predictFactor(np.array(self._dataset[['pu_lon','pu_lat','do_lon','do_lat']]))
		
		self._dataset['grid_trip_dist'] = pd.DataFrame(ratios)
		self._dataset.grid_trip_dist = self._dataset.grid_trip_dist * self._dataset.vec_dist

		cur_df = self._dataset.grid_trip_dist
		return pd.concat([inner_df, cur_df], axis = 1)