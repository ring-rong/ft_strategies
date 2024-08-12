from freqtrade.strategy import IStrategy
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class RingRongv5(IStrategy):
    timeframe = "5m"
    can_short: bool = True

    minimal_roi = {
        "120": 0.20,
        "60": 0.10,
        "0": 0.05
    }

    stoploss = -0.10
    use_custom_stoploss = True
    startup_candle_count = 200

    informative_timeframes = {
        "15m": "15m",
        "1h": "1h",
        "4h": "4h"
    }

    def informative_pairs(self):
        pairs = ["BTC/USDT", "ETH/USDT"]
        informative_pairs = [(pair, tf) for pair in pairs for tf in self.informative_timeframes.values()]
        informative_pairs.append(("BTC/USDT", "5m"))
        informative_pairs.append(("ETH/USDT", "5m"))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Main timeframe indicators (5m)
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

        # Informative timeframes
        for tf_name, tf in self.informative_timeframes.items():
            for pair in ["BTC/USDT", "ETH/USDT"]:
                informative = self.dp.get_pair_dataframe(pair, tf)
                informative[f"ema_{tf}"] = ta.EMA(informative, timeperiod=50)
                informative[f"sma_200_{tf}"] = ta.SMA(informative, timeperiod=200)

                dataframe[f"trend_{pair}_{tf}"] = (
                    informative["close"] > informative[f"sma_200_{tf}"]
                )
                dataframe[f"trend_strength_{pair}_{tf}"] = (
                    (informative["close"] > informative[f"ema_{tf}"])
                    & (informative[f"ema_{tf}"] > informative[f"sma_200_{tf}"])
                )

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
                & (dataframe["trend_BTC/USDT_4h"] == True)
                & (dataframe["trend_strength_BTC/USDT_1h"] == True)
                & (dataframe["volume"] > 0)
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
                & (dataframe["trend_BTC/USDT_4h"] == False)
                & (dataframe["trend_strength_BTC/USDT_1h"] == False)
                & (dataframe["volume"] > 0)
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
            )
            & (dataframe["volume"] > 0),
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
            )
            & (dataframe["volume"] > 0),
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
