import pandas as pd
import os
import numpy as np
import random
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.vector_ar.vecm import VAR
from statsmodels.tsa.stattools import adfuller, coint
from utils import Utils as Ut
from config import Config, logger

#Configuration...
config = Config()
path = Config.get_path_to_data

#Input data...
ticker1 = "AZN"
ticker2 = "PFE"
start_time = "2004-06-16"
end_time = "2010-06-16"
forecast_days = 15
INITIAL_INVESTMENT = 1000


# Forecast-based statistical arbitrage


'''1. Load data'''

pair_paths = Ut.get_data(ticker1, ticker2, path)
unclean_price_series1, unclean_price_series2 = Ut.open_pair_data(ticker1, ticker2, pair_paths)


'''2. Clean data'''

# price_series1_whole_nodiff = Ut.clean_df_nodiff(unclean_price_series1)
# price_series2_whole_nodiff = Ut.clean_df_nodiff(unclean_price_series2)

price_series1_whole = Ut.clean_df(unclean_price_series1)
price_series2_whole = Ut.clean_df(unclean_price_series2)
price_series1_whole_nodiff = Ut.clean_df_nodiff(unclean_price_series1)
price_series2_whole_nodiff =  Ut.clean_df_nodiff(unclean_price_series2)

price_series1 = Ut.create_training_set(price_series1_whole, start_time, end_time)
price_series2 = Ut.create_training_set(price_series2_whole, start_time, end_time)

# print(ticker1 + "\n", price_series1)
# print(ticker2 + "\n", price_series2)


'''3. Cointegration test'''

coint_ratio = Ut.johansen_cointegration(price_series1, price_series2)

logger.info(f"Cointegration test with window")
# some check for coint_ratio of more than x amount.


'''4. Unit root tests'''

#ADF
result = adfuller(price_series1)
result2 = adfuller(price_series2)

# print("p val", result[1])
# print("p val", result2[1])
# p val 1.6410552858534234e-21
# p val 2.0682275928364335e-10

#Phillips Perron
pp1 = PhillipsPerron(price_series1, trend='n')
# print(ticker1 + "...\n", str(pp1.summary()), "\n")
pp2 = PhillipsPerron(price_series2, trend='n')
# print(ticker2 + "...\n", str(pp2.summary()))



'''5. Picking appropriate Lag order'''

lag_candidates = set()

merged_series = Ut.create_merged_series(price_series1, price_series2, ticker1, ticker2)

#Pearson correlation
lags_to_check = 40
pearson_results = Ut.pearson_corr_lags(merged_series, lags_to_check)
lag_candidates.add(key for key in pearson_results.keys())

#VAR tests
result1, result2 = Ut.test_lag_orders_var(merged_series)
result_intersect = np.intersect1d(result1, result2)
lag_candidates.add(result for result in result_intersect)

logger.info(f"Lag candidates: {lag_candidates}")

lag_order = random.choice(list(lag_candidates)) #Needs to be changed lol. For testing purposes only.

logger.info(f"Chose lag order: {lag_order}")


'''6. Forecast'''

forecasted_period = Ut.get_forecast_dates(forecast_days, end_time)
actual1 = Ut.reveal_actual(price_series1_whole_nodiff, forecasted_period['start'], forecasted_period['end'])
actual2 = Ut.reveal_actual(price_series2_whole_nodiff, forecasted_period['start'], forecasted_period['end'])
price_series_whole = Ut.create_merged_series(price_series1_whole, price_series2, ticker1="AZN (whole)", ticker2="PFE (whole)")
forecasted = Ut.forecasted(lag_order, merged_series, forecast_days)

actual_final = Ut.create_merged_series(actual1, actual2, ticker1, ticker2)
forecasted_final = Ut.reverse_differencing(forecasted, price_series_whole, ticker1, ticker2)

'''7. Trade'''

# Buy signal - If asset 1 is predicted to increase in price, whilst the other is predicted to decrease. Then go long.
# Sell signal - If asset 2 is predicted to fall in price, whilst the other is predicted to increase. Then short it.

# Basically if price of both stocks are predicted to diverge in opposite directions.

# Represents the average distance between each value and the average difference 
std_volatility = np.std(price_series1_whole)
std_volatility = np.std(price_series2_whole)
