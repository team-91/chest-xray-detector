# weighted_dataset.py
import numpy as np
from ultralytics.data.dataset import YOLODataset


class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, data=None, task="train", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)

        if task != "train":
            return  

        self.count_instances()
        class_weights = np.sqrt(np.sum(self.counts) / self.counts)
        self.class_weights = np.array(class_weights, dtype=np.float32)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()