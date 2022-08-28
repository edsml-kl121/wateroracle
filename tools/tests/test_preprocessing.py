import tools.preprocessing as preprocessing
import pytest

# B3 = 4
# B5 = 5


@pytest.mark.parametrize('num, result', [
    (13, "12"),
    (7, "07")
])
def test_EnsureTwoDigits(num, result):
    answer = preprocessing.EnsureTwodigit(num)
    assert answer == result


start = {"year": 2018, "month": 1}
end = {"year": 2018, "month": 12}
result1 = ["2018-01-01", "2018-04-01", "2018-07-01", "2018-10-01"]
result2 = ["2018-04-01", "2018-07-01", "2018-10-01", "2018-12-01"]


@pytest.mark.parametrize('start, end, result', [
    (start, end, (result1, result2))
])
def test_GenSeasonalDatesMonthy(start, end, result):
    answer = preprocessing.GenSeasonalDatesMonthly(start, end)
    assert answer == result
