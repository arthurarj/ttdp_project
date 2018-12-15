import numpy as np
import pandas as pd
import abc

# Component (Abstract)
class FeatureExtractorComponent(metaclass=abc.ABCMeta):
    """
    Define the interface for objects that can have responsibilities
    added to them dynamically.
    """

    @abc.abstractmethod
    def getFeatures(self):
        pass

# Concrete Component (the one we instantiate)
class FeatureExtractor(FeatureExtractorComponent):
    """
    Define an object to which additional responsibilities can be
    attached.
    """
    def __init__(self, dataset):
        self._dataset = dataset
        self._feat_list = []

    def getFeatures(self):
        return pd.DataFrame()

# Decorator (Abstract)
class FeatureDecorator(FeatureExtractorComponent, metaclass=abc.ABCMeta):
    """
    Maintain a reference to a Component object and define an interface
    that conforms to Component's interface.
    """

    def __init__(self, component):
        self._component = component
        self._dataset = component._dataset
        self._feat_list = component._feat_list

    @abc.abstractmethod
    def getFeatures(self):
        self._component.getFeatures()