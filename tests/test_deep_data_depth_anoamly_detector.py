import unittest
import numpy as np
from deep_data_depth.DeepDataDepthAnomalyDetector import DeepDataDepthAnomalyDetector


class TestDeepDataDepthAnomalyDetector(unittest.TestCase):
    def test_deep_data_depth_anomaly_detector(self):
        anomaly_detector = DeepDataDepthAnomalyDetector()
        x = np.zeros((5, 3))
        y = np.zeros(5)
        anomaly_detector.fit(x, y)
        self.assertEqual(anomaly_detector.predict_score(x).shape[0], 5)
