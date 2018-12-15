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

class FeatureExtractor(FeatureExtractorComponent):
    """
    Define an object to which additional responsibilities can be
    attached.
    """
    def __init__(self, dataset):
    	self.dataset = dataset

    def getFeatures(self):
        return -1

# Decorator (Abstract)
class FeatureDecorator(FeatureExtractorComponent, metaclass=abc.ABCMeta):
    """
    Maintain a reference to a Component object and define an interface
    that conforms to Component's interface.
    """

    def __init__(self, component):
        self._component = component

    @abc.abstractmethod
    def getFeatures(self):
        self._component.getFeatures()