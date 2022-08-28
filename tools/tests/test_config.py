import tools.config as config
import pytest

train_size = 72
eval_size = 72


@pytest.mark.parametrize('train_size, eval_size', [
    (train_size, eval_size),
])
def test_L8SR(train_size, eval_size):
    storage_config = config.configuration("L8SR",
                                          ["B2", "B3", "B4", "B5", "B6", "B7"],
                                          train_size,
                                          eval_size,
                                          country="global")

    assert storage_config.TRAIN_SIZE == 72
    assert storage_config.EVAL_SIZE == 72
    assert storage_config.PROJECT_TITLE == "L8SR"
    assert storage_config.BANDS1 == ["B2", "B3", "B4", "B5", "B6", "B7"]
    assert storage_config.BANDS == ["B2", "B3", "B4", "B5", "B6", "B7"]
    assert storage_config.country == "global"
    assert storage_config.BUFFER_SIZE == 2000
