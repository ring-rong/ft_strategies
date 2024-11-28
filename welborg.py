import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import matplotlib.pyplot as plt
from freqtrade.strategy import IStrategy, informative, merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from freqtrade.exchange import timeframe_to_prev_date
from technical.indicators import RMI, zema, ichimoku

class VolumVIDYARecurv4(IStrategy):
    """
    Стратегия, основанная на Volume-Weighted VIDYA, зонах ликвидности и WELBORG VERSION 4.

    Эта стратегия использует индикаторы Volume-Weighted VIDYA, зоны ликвидности на основе пивотов
    и сигналы от стратегии WELBORG VERSION 4 для определения точек входа и выхода.
    """

    # Настройки Volume-Weighted VIDYA
    vidya_length = 1
    vidya_momentum = 4
    band_distance = 1.0

    # Настройки зон ликвидности
    pivot_left_bars = 3
    pivot_right_bars = 3
    
    # Цвета для трендов
    up_trend_color = '#17dfad'
    down_trend_color = '#dd326b'
    shadow = True
    
    # Таймфрейм
    timeframe = '5m'
    inf_timeframe = '1h'

    # Отображение сигналов от стратегии WELBORG VERSION 4
    welborg_show_signals = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Функция для расчета индикаторов.
        """
        # Вычисление Average True Range (ATR)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=200)
        
        # Вычисление Volume-Weighted VIDYA
        dataframe = self.calculate_vidya(dataframe)
        
        # Вычисление верхней и нижней полос на основе VIDYA и ATR
        dataframe['upper_band'] = dataframe['vidya'] + dataframe['atr'] * self.band_distance
        dataframe['lower_band'] = dataframe['vidya'] - dataframe['atr'] * self.band_distance

        # Определение направления тренда с помощью пересечений полос
        dataframe['is_trend_up'] = (
            (qtpylib.crossed_above(dataframe['close'], dataframe['upper_band'])) |  
            (dataframe['is_trend_up'].shift(1) & (dataframe['close'] > dataframe['lower_band']))
        ).astype(int)
        
        # Настройка переменной сглаживания на основе тренда
        dataframe['smoothed_value'] = np.where(dataframe['is_trend_up'], 
                                               dataframe['lower_band'], 
                                               dataframe['upper_band'])
        dataframe.loc[dataframe['is_trend_up'].shift(1) != dataframe['is_trend_up'], 'smoothed_value'] = np.nan
        
        # Расчет пивотов для цены
        dataframe['pivot_high'] = ta.pivothigh(dataframe['high'], self.pivot_left_bars, self.pivot_right_bars)
        dataframe['pivot_low'] = ta.pivotlow(dataframe['low'], self.pivot_left_bars, self.pivot_right_bars)

        # Расчет сигналов от стратегии WELBORG VERSION 4
        if self.welborg_show_signals:
            welborg_signals = self.populate_welborg_signals(dataframe)
            dataframe = pd.concat([dataframe, welborg_signals], axis=1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Функция для определения точек входа.
        """
        dataframe.loc[
            (dataframe['close'] > dataframe['upper_band']) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=50).mean()),
            'enter_long'] = 1

        dataframe.loc[
            (dataframe['close'] < dataframe['lower_band']) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=50).mean()),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Функция для определения точек выхода.
        """
        dataframe.loc[
            (dataframe['close'] < dataframe['smoothed_value']) & 
            qtpylib.crossed_below(dataframe['close'], dataframe['lower_band']),
            'exit_long'] = 1
        
        dataframe.loc[
            (dataframe['close'] > dataframe['smoothed_value']) & 
            qtpylib.crossed_above(dataframe['close'], dataframe['upper_band']),
            'exit_short'] = 1

        return dataframe

    def calculate_vidya(self, dataframe: DataFrame) -> DataFrame:
        """
        Функция для вычисления Volume-Weighted VIDYA.
        """
        momentum = dataframe['close'].diff()
        sum_pos_momentum = momentum.rolling(window=self.vidya_momentum).apply(lambda x: x[x >= 0].sum(), raw=True)
        sum_neg_momentum = momentum.rolling(window=self.vidya_momentum).apply(lambda x: -x[x < 0].sum(), raw=True)
        abs_cmo = 100 * abs(sum_pos_momentum - sum_neg_momentum) / (sum_pos_momentum + sum_neg_momentum)
        alpha = 2 / (self.vidya_length + 1)
        
        vidya_value = abs_cmo / 100 * dataframe['close']
        vidya_value = alpha * vidya_value + (1 - alpha * abs_cmo / 100) * vidya_value.shift(1)
        vidya_value = vidya_value.rolling(window=15).mean()

        dataframe['vidya'] = vidya_value

        return dataframe
    
    def populate_welborg_signals(self, dataframe: DataFrame) -> DataFrame:
        """
        Функция для расчета сигналов от стратегии WELBORG VERSION 4.
        """
        # Настройки Smooth Average Range
        per1 = 27
        mult1 = 1.6
        per2 = 55
        mult2 = 2.0

        # Вычисление Smooth Average Range
        def smoothrng(x, t, m):
            wper = t * 2 - 1
            avrng = ta.EMA(abs(x - x.shift(1)), t)
            smoothrng = ta.EMA(avrng, wper) * m
            return smoothrng

        smrng1 = smoothrng(dataframe['close'], per1, mult1)
        smrng2 = smoothrng(dataframe['close'], per2, mult2)
        smrng = (smrng1 + smrng2) / 2

        # Range Filter
        def rngfilt(x, r):
            rngfilt = pd.Series(np.nan, index=x.index)
            for i in range(1, len(x)):
                if x.iloc[i] > rngfilt.iloc[i-1]:
                    rngfilt.iloc[i] = min(x.iloc[i] - r, rngfilt.iloc[i-1])
                else:
                    rngfilt.iloc[i] = max(x.iloc[i] + r, rngfilt.iloc[i-1])
            return rngfilt

        filt = rngfilt(dataframe['close'], smrng)

        # Расчет направления тренда
        upward = pd.Series(np.nan, index=dataframe.index)
        downward = pd.Series(np.nan, index=dataframe.index)
        for i in range(1, len(filt)):
            if filt.iloc[i] > filt.iloc[i-1]:
                upward.iloc[i] = upward.iloc[i-1] + 1 if not np.isnan(upward.iloc[i-1]) else 1
                downward.iloc[i] = 0
            elif filt.iloc[i] < filt.iloc[i-1]:
                downward.iloc[i] = downward.iloc[i-1] + 1 if not np.isnan(downward.iloc[i-1]) else 1
                upward.iloc[i] = 0
            else:
                upward.iloc[i] = upward.iloc[i-1]
                downward.iloc[i] = downward.iloc[i-1]

        hband = filt + smrng
        lband = filt - smrng

        # Определение условий для длинных и коротких позиций
        longCond = ((dataframe['close'] > filt) & (dataframe['close'] > dataframe['close'].shift(1)) & (upward > 0)) | \
                   ((dataframe['close'] > filt) & (dataframe['close'] < dataframe['close'].shift(1)) & (upward > 0))
        
        shortCond = ((dataframe['close'] < filt) & (dataframe['close'] < dataframe['close'].shift(1)) & (downward > 0)) | \
                    ((dataframe['close'] < filt) & (dataframe['close'] > dataframe['close'].shift(1)) & (downward > 0))

        CondIni = pd.Series(np.nan, index=dataframe.index)
        for i in range(1, len(dataframe)):
            if longCond.iloc[i]:
                CondIni.iloc[i] = 1
            elif shortCond.iloc[i]:
                CondIni.iloc[i] = -1
            else:
                CondIni.iloc[i] = CondIni.iloc[i-1]

        # Сигналы на покупку и продажу
        buy_signal = (longCond) & (CondIni.shift(1) == -1)
        sell_signal = (shortCond) & (CondIni.shift(1) == 1)

        signals = pd.concat([buy_signal.rename('welborg_buy'), sell_signal.rename('welborg_sell')], axis=1)

        return signals

    def plot_config(self, df):
        """
        Функция для настройки графика.
        """
        plot_config = {
            'main_plot': {
                'vidya': {'color': 'green'},
                'smoothed_value': {'color': 'orange'},
                'upper_band': {'color': 'red'},
                'lower_band': {'color': 'blue'},
                'pivot_high': {'color': 'purple'},
                'pivot_low': {'color': 'purple'},
                'welborg_buy': {'type': 'scatter', 'color': 'green', 'marker': '^'},
                'welborg_sell': {'type': 'scatter', 'color': 'red', 'marker': 'v'},
            },
            'subplots': {
                'volume': {'color': 'blue'}
            }
        }
        return plot_config

    def informative_pairs(self):
        """
        Функция для определения дополнительных пар для получения информации.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return informative_pairs
