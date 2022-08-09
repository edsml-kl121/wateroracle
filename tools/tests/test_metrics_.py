import tools.metrics_ as metrics
import pytest
import tensorflow as tf


test_data = [(tf.random.uniform(shape=(1,256,256,2)), tf.zeros(shape=(1,256,256,2)))]
total_steps = 1


@pytest.mark.parametrize('test_data, total_steps', [
    # (torch.randn((5,366,366)).numpy(),'./images/Real'), # from root of project
    (test_data, 1),
])
def test_Metrics_NDWI(test_data, total_steps):
    precision_macro, recall_macro, f1_macro, accuracy = metrics.MetricCalculator_NDWI(test_data, total_steps)
    assert precision_macro <= 0 or precision_macro >= 1
    assert recall_macro <= 0 or precision_macro >= 1
    assert f1_macro <= 0 or precision_macro >= 1
    assert accuracy <= 0 or precision_macro >= 1


B3 = 4
B5 = 5

@pytest.mark.parametrize('B3, B5', [
    # (torch.randn((5,366,366)).numpy(),'./images/Real'), # from root of project
    (B3, B5),
])
def test_NDWI(B3, B5):
    first, second = metrics.ndwi_threashold(B3, B5)
    assert first == 0 or first == 1
    assert second == 0 or second == 1
