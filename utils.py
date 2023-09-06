import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats import pearsonr
from datetime import datetime, timedelta
from config import logger
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import VAR


class Utils():

    @staticmethod
    def plot_df(price_series1, price_series2=None, merge_and_plot=False, title="", xlabel="", ylabel="", color="blue"):
        plt.style.use('default')
        if merge_and_plot and price_series2 is not None:
            plt.plot(price_series1, color="blue", linewidth=0.75, label=f"")
            plt.plot(price_series2, color="green", linewidth=0.75, label=f"")
            plt.legend()
        else:
            plt.plot(price_series1, color, linewidth=0.75)
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        year_locator = mdates.YearLocator(base=1)
        plt.gca().xaxis.set_major_locator(year_locator)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def difference(price_series):
        new_series = price_series - price_series.shift(1)
        new_series = new_series.dropna()
        new_index = price_series.index[1:]
        new_series.index = new_index
        return new_series
    
    @staticmethod
    def get_data(ticker1, ticker2, path_to_data):
        pair_paths = dict()
        for root, dirs, files in os.walk(path_to_data):
            for name in files:
                if name == f"{ticker1}.csv":
                    ticker1_path = str(os.path.join(root, name))
                    pair_paths[ticker1] = ticker1_path
                elif name == f"{ticker2}.csv":
                    ticker2_path = str(os.path.join(root, name))
                    pair_paths[ticker2] = ticker2_path
        return pair_paths
    
    @staticmethod
    def clean_df(unclean_price_series):
        price_df = unclean_price_series.set_index('datetime')
        price_df.index = pd.to_datetime(price_df.index)
        new_price_df = price_df['close'].copy()
        new_price_df = Utils.difference(new_price_df)
        new_price_df.index.asfreq = 'B'
        return new_price_df
    
    @staticmethod
    def clean_df_nodiff(unclean_price_series):
        price_df = unclean_price_series.set_index('datetime')
        price_df.index = pd.to_datetime(price_df.index)
        new_price_df = price_df['close'].copy()
        new_price_df.index.asfreq = 'B'
        return new_price_df
    
    @staticmethod
    def plot_and_extract_pacf(price_df, title, lags, plot=False):
        fig, ax = plt.subplots(figsize=(8, 6))
        y_limit = 0.9
        plot_pacf(price_df, lags=lags, ax=ax)
        ax.set_title(title)
        ax.set_ylim(-y_limit, y_limit)
        tick_positions = np.arange(0, lags + 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, rotation=45, ha='right', fontsize=5.5)
        plt.tight_layout()
        plt.show()
        return 

    @staticmethod
    def pearson_corr_lags(merged_series, lags):
        result = {}
        for lag in range(1, lags+1):
            price_series1 = merged_series['price1'].iloc[lag:]
            lagged_price_series2 = merged_series['price2'].iloc[:-lag]
            vals = pearsonr(price_series1, lagged_price_series2)
            
            if vals[1] <= 0.01:
                result[f'{lag}'] = vals[1]
        return result
    
    @staticmethod
    def create_merged_series(price_series1, price_series2, ticker1, ticker2):
        merged_series = pd.DataFrame()
        if all(price_series1.index == price_series2.index):
            merged_series.index = price_series1.index
            merged_series[f"{ticker1}"] = price_series1
            merged_series[f"{ticker2}"] = price_series2
            merged_series.index.asfreq = 'B'
        else:
            raise ValueError("The index of price_series1 and price_series2 do not match.")

        return merged_series
    
    @staticmethod
    def check_datetime_index_consistency(series):
        if not isinstance   (series.index, pd.DatetimeIndex):
            raise ValueError("The input series should have a datetime index.")
        index_values = series.index
        time_diffs = index_values[1:] - index_values[:-1]
        consistency = all(time_diff == time_diffs[0] for time_diff in time_diffs)
        if not consistency:
            total_days_skipped = (index_values[-1] - index_values[0]).days - len(index_values) + 1
        else:
            total_days_skipped = None
        missing = index_values[index_values.to_series().diff() > pd.Timedelta(days=1)]
        duplicates = index_values[index_values.duplicated()]
        result = {
            "consistent_format": consistency,
            "total_days_skipped": total_days_skipped,
            "missing_values" : missing,
            "duplicate_values" : duplicates}
        print(result)
    
    @staticmethod
    def get_business_days(start_date, end_date):
        business_days = np.busday_count(start_date, end_date)
        return business_days
    
    @staticmethod
    def get_forecast_dates(forecast_days, end_time):
        forecast_start_time_dt = datetime.strptime(end_time, "%Y-%m-%d") + timedelta(days=1)
        forecast_start_time = forecast_start_time_dt.strftime("%Y-%m-%d")
        forecast_end_time_dt = Utils.date_by_adding_business_days(forecast_start_time_dt, forecast_days)
        forecast_end_time = forecast_end_time_dt.strftime("%Y-%m-%d")
        business_days = Utils.get_business_days(forecast_start_time, forecast_end_time)
        return {"start": forecast_start_time, "end": forecast_end_time, "num_biz_days" : business_days}
    
    @staticmethod
    def date_by_adding_business_days(from_date, add_days):
        business_days_to_add = add_days
        current_date = from_date
        while business_days_to_add > 0:
            current_date += timedelta(days=1)
            weekday = current_date.weekday()
            if weekday >= 5: # sunday = 6
                continue
            business_days_to_add -= 1
        return current_date
    
    @staticmethod
    def open_pair_data(ticker1, ticker2, pair_paths):
        with open(rf"{pair_paths[ticker1]}", "r") as s1:
            with open(rf"{pair_paths[ticker2]}", "r") as s2:
                unclean_price_series1 = pd.read_csv(s1)
                logger.info(ticker1 + "\n", unclean_price_series1)

                unclean_price_series2 = pd.read_csv(s2)
                print(ticker2 + "\n", unclean_price_series2)
                return unclean_price_series1, unclean_price_series2
    
    @staticmethod
    def create_training_set(price_series, start_time, end_time):
        new_price_series = price_series
        new_price_series = new_price_series.truncate(start_time, end_time)
        return new_price_series
    
    @staticmethod
    def johansen_cointegration(price_series1, price_series2):
        window_size = 35
        cointegration_results = []
        for i in range(len(price_series1) - window_size + 1):
            window_price_series1 = price_series1[i:i+window_size]
            window_price_series2 = price_series2[i:i+window_size]
            score, p_value, _ = coint(window_price_series1, window_price_series2)
            cointegration_results.append(p_value < 0.05)

        num_cointegrated = sum(cointegration_results)
        coint_ratio = num_cointegrated/len(cointegration_results)
        return coint_ratio
    
    @staticmethod
    def test_lag_orders_var(merged_series):
        model = VAR(merged_series)
        model_fit = model.fit(maxlags=20)
        p_values = model_fit.pvalues
        result1 = []
        result2 = []
        for lag, val in p_values['price1'].items():
            if val <= 0.05:
                logger.info(f"{lag}: p-value = {round(val, 3)}")
                result1.append(lag)
        for lag, val in p_values['price2'].items():
            if val <= 0.05:
                logger.info(f"{lag}: p-value = {round(val, 3)}")
                result2.append(lag)
        return np.array(result1), np.array(result2)
    
    @staticmethod
    def forecasted(lag_order, merged_series, forecast_days):
        forecast_days = forecast_days + 1
        model = VAR(merged_series)
        model_fit = model.fit(lag_order)
        last_lagged_observations = merged_series[-lag_order:]
        forecast = model_fit.forecast(last_lagged_observations.values, steps=forecast_days)
        forecast_df = pd.DataFrame(forecast, columns=merged_series.columns)
        forecast_dates = pd.date_range(start=merged_series.index[0] + pd.DateOffset(days=1), periods=forecast_days, freq='B')
        forecast_df.index = forecast_dates
        forecast_df = forecast_df[::-1]
        return forecast_df

    @staticmethod
    def reveal_actual(whole_price_series, forecast_start_time, forecast_end_time):
        actual_series = whole_price_series.truncate(forecast_start_time, forecast_end_time)
        return actual_series
    
    @staticmethod
    def reverse_differencing(forecasted, price_series_whole, ticker1, ticker2):
        def _generate_col(new_df, current_forecast, ticker, last_known):
            if current_forecast == forecasted.index[-1]:
                return new_df
            
            difference = forecasted[f"{ticker}"][current_forecast]
            new_df[f'{ticker} (forecasted)'][current_forecast] = last_known + difference
            last_known = new_df[f'{ticker} (forecasted)'][current_forecast]
            current_forecast += timedelta(days=1)
            return _generate_col(new_df, current_forecast, ticker, last_known)
        
        new_df = pd.DataFrame()
        new_df.index = forecasted.index
        current_forecast = forecasted.index[0]

        new_df[f'{ticker1} (forecasted)'] = None
        last_known1 = price_series_whole.loc[forecasted.index[0] - timedelta(days=1), f"{ticker1} (whole)"]
        new_df = _generate_col(new_df, current_forecast, ticker1, last_known1)

        new_df[f'{ticker2} (forecasted)'] = None
        last_known2 = price_series_whole.loc[forecasted.index[0] - timedelta(days=1), f"{ticker2} (whole)"]
        new_df = _generate_col(new_df, current_forecast, ticker2, last_known2)

        return new_df
    
    @staticmethod
    def calc_lag_std(price_series, lag_order):
        aggregated_means = []
        for i in range(0, len(price_series) - lag_order + 1, lag_order):
            aggregated_mean = np.mean(price_series[i:i + lag_order])
            aggregated_means.append(aggregated_mean)

        std_deviation = np.std(aggregated_means)
        return std_deviation

    @staticmethod
    def find_historical_differences(price_series1, price_series2):
        if len(price_series1) == len(price_series2):
            new_series = price_series1
            new_series = price_series1 - price_series2
            return new_series
        else:
            raise Exception("Both price series must be of the same length")

        
        

    
        
    
    # @staticmethod
    # def find_pacf_significant_lags(pacf_values, conf_intervals, threshold=0.1):
    #     """
    #     Find lag orders with significant partial autocorrelation values based on confidence intervals.

    #     Parameters:
    #         pacf_values (ndarray): Partial autocorrelation values.
    #         conf_intervals (ndarray): Confidence intervals for PACF values.
    #         threshold (float, default=0.1): Threshold for identifying significant lags.

    #     Returns:
    #         significant_lags (list): List of lag orders with significant partial autocorrelation.
    #     """
    #     significant_lags = []
    #     for lag, pacf_value, conf_interval in zip(range(len(pacf_values)), pacf_values, conf_intervals):
    #         lower_bound, upper_bound = conf_interval
    #         if abs(pacf_value) > threshold and pacf_value > upper_bound:
    #             significant_lags.append(lag)
    #     return significant_lags