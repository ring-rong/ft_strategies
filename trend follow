from freqtrade.strategy import IStrategy
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class RingRongTrendFollowing(IStrategy):
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
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["macd"], dataframe["macdsignal"], dataframe["macdhist"] = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # Informative timeframes
        for tf_name, tf in self.informative_timeframes.items():
            for pair in ["BTC/USDT", "ETH/USDT"]:
                informative = self.dp.get_pair_dataframe(pair, tf)
                informative[f"ema_50_{tf}"] = ta.EMA(informative, timeperiod=50)
                informative[f"ema_200_{tf}"] = ta.EMA(informative, timeperiod=200)
                informative[f"rsi_{tf}"] = ta.RSI(informative, timeperiod=14)
                informative[f"macd_{tf}"], informative[f"macdsignal_{tf}"], _ = ta.MACD(informative, fastperiod=12, slowperiod=26, signalperiod=9)

                # Add trend determination based on EMA crossovers
                dataframe[f"trend_{pair}_{tf}"] = (
                    informative[f"ema_50_{tf}"] > informative[f"ema_200_{tf}"]
                )
                dataframe[f"trend_strength_{pair}_{tf}"] = (
                    (informative[f"ema_50_{tf}"] > informative[f"ema_200_{tf}"])
                    & (informative[f"macd_{tf}"] > informative[f"macdsignal_{tf}"])
                )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Entry: Follow the higher timeframe trend
        dataframe.loc[
            (
                (dataframe["trend_BTC/USDT_4h"] == True)  # Overall market uptrend on 4h
                & (dataframe["trend_strength_BTC/USDT_1h"] == True)  # Strong trend on 1h
                & (dataframe["ema_50"] > dataframe["ema_200"])  # Confirmed uptrend on 5m
                & (dataframe["rsi"] > 50)  # RSI indicates bullish momentum
                & (dataframe["macd"] > dataframe["macdsignal"])  # MACD bullish crossover
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "trend_follow_long")

        # Short Entry: Follow the higher timeframe trend
        dataframe.loc[
            (
                (dataframe["trend_BTC/USDT_4h"] == False)  # Overall market downtrend on 4h
                & (dataframe["trend_strength_BTC/USDT_1h"] == False)  # Strong downtrend on 1h
                & (dataframe["ema_50"] < dataframe["ema_200"])  # Confirmed downtrend on 5m
                & (dataframe["rsi"] < 50)  # RSI indicates bearish momentum
                & (dataframe["macd"] < dataframe["macdsignal"])  # MACD bearish crossover
                & (dataframe["volume"] > 0)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "trend_follow_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit Long: Market shows signs of trend reversal
        dataframe.loc[
            (
                (dataframe["rsi"] < 50)  # RSI indicates weakening bullish momentum
                | (qtpylib.crossed_below(dataframe["ema_50"], dataframe["ema_200"]))  # EMA crossover to bearish
                | (dataframe["macd"] < dataframe["macdsignal"])  # MACD bearish crossover
                | (dataframe["atr"] < 0.005)  # Low ATR indicates low volatility
            )
            & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "trend_reversal_long")

        # Exit Short: Market shows signs of trend reversal
        dataframe.loc[
            (
                (dataframe["rsi"] > 50)  # RSI indicates weakening bearish momentum
                | (qtpylib.crossed_above(dataframe["ema_50"], dataframe["ema_200"]))  # EMA crossover to bullish
                | (dataframe["macd"] > dataframe["macdsignal"])  # MACD bullish crossover
                | (dataframe["atr"] < 0.005)  # Low ATR indicates low volatility
            )
            & (dataframe["volume"] > 0),
            ["exit_short", "exit_tag"],
        ] = (1, "trend_reversal_short")

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
