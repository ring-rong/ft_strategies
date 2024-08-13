from freqtrade.strategy import IStrategy
from pandas import DataFrame
from datetime import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

class RingRong(IStrategy):
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

        # Count consecutive green/red candles
        dataframe['consec_green'] = (dataframe['close'] > dataframe['open']).astype(int).groupby((dataframe['close'] <= dataframe['open']).astype(int).cumsum()).cumsum()
        dataframe['consec_red'] = (dataframe['close'] < dataframe['open']).astype(int).groupby((dataframe['close'] >= dataframe['open']).astype(int).cumsum()).cumsum()

        # Calculate candle shadows (wicks/tails)
        dataframe['upper_shadow'] = dataframe['high'] - dataframe[['close', 'open']].max(axis=1)
        dataframe['lower_shadow'] = dataframe[['close', 'open']].min(axis=1) - dataframe['low']

        # Define significant shadow (wick/tail) threshold
        dataframe['body_size'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['is_long_upper_shadow'] = dataframe['upper_shadow'] > 1.5 * dataframe['body_size']
        dataframe['is_long_lower_shadow'] = dataframe['lower_shadow'] > 1.5 * dataframe['body_size']

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
                & (dataframe['consec_red'] >= 3)  # 3 or more consecutive red candles
                & (dataframe["rsi"] < 30)  # RSI is oversold
                & dataframe['is_long_lower_shadow']  # Significant lower shadow (bullish reversal signal)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "adx_bullish_reversal")

        # Bearish entry conditions
        dataframe.loc[
            (
                (dataframe["adx"] > 25)  # ADX shows stronger trend
                & (dataframe["short"] < dataframe["long"])
                & (dataframe["close"] < dataframe["long"])  # Price below long-term EMA
                & (dataframe["rsi"] < 50)  # RSI below 50 indicates bearish momentum
                & (dataframe["close"] < dataframe["bb_middleband"])  # Price below middle BB
                & (
                    (dataframe['consec_green'] >= 3)  # 3 or more consecutive green candles
                    | (dataframe["rsi"] > 70)  # RSI is overbought
                )
                & dataframe['is_long_upper_shadow']  # Significant upper shadow (bearish reversal signal)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "adx_bearish_reversal")

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
