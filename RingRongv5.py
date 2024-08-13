from freqtrade.strategy import IStrategy
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class RingRong(IStrategy):
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
        # Enter Long: Exit short and enter long
        dataframe.loc[
            (
                (dataframe["adx"] > 25)  # Strong trend
                & (dataframe["short"] > dataframe["long"])
                & (dataframe["close"] > dataframe["long"])  # Above long-term EMA
                & (dataframe["rsi"] > 50)  # Bullish momentum
                & (dataframe["close"] > dataframe["bb_middleband"])  # Above middle BB
                & (dataframe["trend_BTC/USDT_4h"] == True)
                & (dataframe["trend_strength_BTC/USDT_1h"] == True)
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "exit_short", "enter_tag", "exit_tag"],
        ] = (1, 1, "switch_to_long", "exit_short_to_long")

        # Enter Short: Exit long and enter short
        dataframe.loc[
            (
                (dataframe["adx"] > 25)  # Strong trend
                & (dataframe["short"] < dataframe["long"])
                & (dataframe["close"] < dataframe["long"])  # Below long-term EMA
                & (dataframe["rsi"] < 50)  # Bearish momentum
                & (dataframe["close"] < dataframe["bb_middleband"])  # Below middle BB
                & (dataframe["trend_BTC/USDT_4h"] == False)
                & (dataframe["trend_strength_BTC/USDT_1h"] == False)
                & (dataframe["volume"] > 0)
            ),
            ["enter_short", "exit_long", "enter_tag", "exit_tag"],
        ] = (1, 1, "switch_to_short", "exit_long_to_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Since we flip between positions, explicit exit conditions are not required.
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
