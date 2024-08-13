from freqtrade.strategy import IStrategy
from pandas import DataFrame
from datetime import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade  # <-- Make sure this is imported
"""
| Strategy      | Buys | Avg Prof% | Cum Prof% | Tot Prof% | Tot Profit% | Avg Duration | Win | Draw | Loss | Win% | DD Stake | DD%   | Time  | Stake | Exchange | Stoploss | Sharpe | Sortino | Calmar | Month  |
|---------------|------|-----------|-----------|-----------|-------------|--------------|-----|------|------|-------|----------|-------|-------|--------|----------|---------|--------|---------|--------|--------|
| RingRong15m   | 699  | -0.02     | -15.47    | -37.22    | -2.86       | 2:33:00      | 207 | 0    | 492  | 29.6  | 249.908  | 17.12 | 64.7  | USDT  | binance  | -4.54   | -4.54  | -10.67  | -10.3  | 202407 |
| RingRong15m   | 694  | 0.02      | 12.33     | 39.54     | 3.04        | 2:22:00      | 224 | 0    | 470  | 32.3  | 164.777  | 12.05 | 61.1  | USDT  | binance  | 4.71    | 4.71   | 9.5     | 16.07  | 202406 |
| RingRong15m   | 782  | -0.08     | -65.5     | -190.58   | -14.66      | 2:09:00      | 241 | 0    | 541  | 30.8  | 295.688  | 21.09 | 61.7  | USDT  | binance  | -24.82  | -24.82 | -46.91  | -42.84 | 202405 |
| RingRong15m   | 709  | 0.03      | 19.69     | 47.38     | 3.64        | 2:08:00      | 232 | 0    | 477  | 32.7  | 300.526  | 18.67 | 59.6  | USDT  | binance  | 4.74    | 4.74   | 11.04   | 12.43  | 202404 |
| RingRong15m   | 986  | -0.15     | -145.77   | -412.43   | -31.73      | 1:41:00      | 278 | 0    | 708  | 28.2  | 527.332  | 37.31 | 63.7  | USDT  | binance  | -36.79  | -36.79 | -85.52  | -52.4  | 202403 |
| RingRong15m   | 721  | -0.11     | -78.07    | -220.13   | -16.93      | 2:06:00      | 226 | 0    | 495  | 31.3  | 245.746  | 18.63 | 63    | USDT  | binance  | -27.24  | -27.24 | -61.75  | -59.87 | 202402 |
| RingRong15m   | 636  | -0.2      | -127.38   | -382.56   | -29.43      | 2:00:00      | 187 | 0    | 449  | 29.4  | 515.623  | 37.39 | 55.3  | USDT  | binance  | -45.19  | -45.19 | -87.86  | -48.5  | 202401 |
| RingRong15m   | 854  | -0.02     | -19.83    | -69.79    | -5.37       | 2:05:00      | 266 | 0    | 588  | 31.1  | 248.456  | 19.1  | 66.2  | USDT  | binance  | -6.57   | -6.57  | -15.23  | -17.32 | 202312 |
| RingRong15m   | 767  | -0.16     | -119.76   | -360.15   | -27.7       | 2:04:00      | 231 | 0    | 536  | 30.1  | 443.521  | 32.07 | 61.1  | USDT  | binance  | -34.89  | -34.89 | -66.58  | -55.01 | 202311 |
| RingRong15m   | 678  | 0.08      | 52.58     | 158.04    | 12.16       | 2:32:00      | 217 | 0    | 461  | 32    | 93.556   | 6.12  | 63.3  | USDT  | binance  | 17.77   | 17.77  | 47.3    | 122.36 | 202310 |
| RingRong15m   | 687  | 0.06      | 37.87     | 114.01    | 8.77        | 2:08:00      | 209 | 0    | 478  | 30.4  | 116.561  | 8.57  | 56.3  | USDT  | binance  | 12.8    | 12.8   | 35.36   | 65.17  | 202309 |
| RingRong15m   | 620  | -0.1      | -63.11    | -188.30   | -14.48      | 2:23:00      | 172 | 0    | 448  | 27.7  | 282.419  | 21.4  | 56    | USDT  | binance  | -19.04  | -19.04 | -33.73  | -41.71 | 202308 |

Summary:

| Metric          | Value     |
|-----------------|-----------|
| Buys            | 704       |
| Avg Prof%       | 0         |
| Tot Prof%       | -9.9      |
| Win%            | 30        |
| DD%             | 18        |
| Time            | 61.3      |
| Sharpe          | -12       |
| Sortino         | -24       |
| Calmar          | -29       |
| Prof. Factor    | 0.9       |
| Expectancy      | -0.1      |
| CAGR            | -0.6      |
| Trades/Day      | 24.54     |
| Rejected Signals| 90        |
| Ninja Score     | 61        |
"""
class RingRong15m(IStrategy):
    timeframe = "15m"
    can_short: bool = True

    minimal_roi = {
        "120": 0.20,
        "60": 0.10,
        "0": 0.05
    }

    stoploss = -0.10
    use_custom_stoploss = True
    startup_candle_count = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators on 15m timeframe
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["short"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["long"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # Volume Moving Average (VMA) for detecting volume spikes
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_mean_fast'] = dataframe['volume'].rolling(window=5).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bullish entry conditions
        dataframe.loc[
            (
                (dataframe["adx"] > 25)  # ADX shows stronger trend
                & (dataframe["short"] > dataframe["long"])
                & (dataframe["close"] > dataframe["long"])  # Price above long-term EMA
                & (dataframe["rsi"] > 50)  # RSI above 50 indicates bullish momentum
                & (dataframe["close"] > dataframe["bb_middleband"])  # Price above middle BB
                & (dataframe['volume'] > 1.5 * dataframe['volume_mean_slow'])  # Volume spike
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "adx_bullish")

        # Bearish entry conditions
        dataframe.loc[
            (
                (dataframe["adx"] > 25)  # ADX shows stronger trend
                & (dataframe["short"] < dataframe["long"])
                & (dataframe["close"] < dataframe["long"])  # Price below long-term EMA
                & (dataframe["rsi"] < 50)  # RSI below 50 indicates bearish momentum
                & (dataframe["close"] < dataframe["bb_middleband"])  # Price below middle BB
                & (dataframe['volume'] > 1.5 * dataframe['volume_mean_slow'])  # Volume spike
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "adx_bearish")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bullish exit conditions
        dataframe.loc[
            (
                (dataframe["adx"] < 20)  # Weakening trend
                | (qtpylib.crossed_below(dataframe["short"], dataframe["long"]))
                | (dataframe["close"] < dataframe["bb_middleband"])  # Price falls below middle BB
                | (dataframe["rsi"] < 50)  # RSI drops below 50
                | (dataframe["atr"] < 0.005)  # Low ATR indicates low volatility
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "exit_long")

        # Bearish exit conditions
        dataframe.loc[
            (
                (dataframe["adx"] < 20)  # Weakening trend
                | (qtpylib.crossed_above(dataframe["short"], dataframe["long"]))
                | (dataframe["close"] > dataframe["bb_middleband"])  # Price rises above middle BB
                | (dataframe["rsi"] > 50)  # RSI rises above 50
                | (dataframe["atr"] < 0.005)  # Low ATR indicates low volatility
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "exit_short")

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Custom stoploss with trailing stop based on ATR.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe['atr'].iloc[-1]

        # Trailing stoploss logic
        if current_profit > 0.05:
            stoploss = max(-0.05, current_profit - atr * 1.5)
        else:
            stoploss = max(-0.10, -1.5 * atr)

        return stoploss
