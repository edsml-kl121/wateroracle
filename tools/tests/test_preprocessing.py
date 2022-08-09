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



# def EnsureTwodigit(number):
#   """
#   Transform the input month into string in the
#   correct format for date and time.
#   ----------
#   number: int

#   Returns
#   ----------
#   months in string.

#   """
#   if number > 12:
#     return str(12)
#   if number < 10:
#     return "0"+str(number)
#   else:
#     return str(number)

# def GenSeasonalDatesMonthly(start, end, month_frequency = 3):
#   """
#   Given two dictionary containing the key month and year,
#   return two arrays that contains the time between the 
#   interval of start and end.
#   ----------
#   start: dict
#   end: dict

#   Returns
#   ----------
#   Two arrays containing the time elapsed between start and end

#   """
#   diff_year = end["year"] - start["year"]
#   diff_month = end["month"] - start["month"]
#   starts = []
#   ends = []
#   first_data = str(start["year"]) + "-" + EnsureTwodigit(start["month"]) + "-01"
#   if diff_year > 0:
#     return "please insert the same year"
#   else:
#     for i in range(round(diff_month/month_frequency)):
#       first_data = str(start["year"]) + "-" + EnsureTwodigit(start["month"] + month_frequency * i) + "-01"
#       second_data = str(start["year"]) + "-" + EnsureTwodigit(start["month"] + month_frequency * i + month_frequency) + "-01"
#       starts.append(first_data)
#       ends.append(second_data)
#   return starts, ends
