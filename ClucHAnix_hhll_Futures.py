from typing import Optional  # Add this line
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame, Series
from datetime import datetime, timedelta
from freqtrade.persistence import Trade


class ClucHAnix_hhll_Futures(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '1m'  # Shorter timeframe for futures

    # Leverage and position sizing
    leverage = IntParameter(1, 20, default=5, space='buy', optimize=True)
    position_size = DecimalParameter(0.01, 1.0, default=0.1, space='buy', optimize=True)

    # Buy hyperspace params
    buy_params = {
        "bbdelta_close": 0.01889,
        "bbdelta_tail": 0.95089,
        "close_bblower": 0.00799,
        "closedelta_close": 0.00556,
        "rocr_1h": 0.54904,
    }

    # Sell hyperspace params
    sell_params = {
        "sell_fisher": 0.38414,
        "sell_bbmiddle_close": 1.07634,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.103,
        "3": 0.05,
        "5": 0.033,
        "61": 0.027,
        "125": 0.011,
        "292": 0.005,
    }

    # Stoploss:
    stoploss = -0.99  # use custom stoploss

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = False

    # Run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'emergencysell': 'market',
        'forcebuy': "market",
        'forcesell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])

        # EMA
        dataframe['ema_fast'] = ta.EMA(dataframe['close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['close'], timeperiod=50)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # ROCR
        dataframe['rocr'] = ta.ROCR(dataframe['close'], timeperiod=28)

        # Fisher Transform
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Funding rate (you'll need to implement this based on your data source)
        # dataframe['funding_rate'] = self.dp.get_funding_rate(metadata['pair'])

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rocr'] > self.buy_params['rocr_1h']) &
                (dataframe['bb_width'] > 0.01) &
                (dataframe['close'] < dataframe['bb_lowerband']) &
                (dataframe['fisher'] < -0.5)
            ),
            'buy'] = 1

        return dataframe

    def populate_short_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rocr'] < -self.buy_params['rocr_1h']) &
                (dataframe['bb_width'] > 0.01) &
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['fisher'] > 0.5)
            ),
            'short'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['fisher'] > self.sell_params['sell_fisher']) &
                (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
                (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
                (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
                (dataframe['ema_fast'] > dataframe['ha_close']) &
                ((dataframe['ha_close'] * self.sell_params['sell_bbmiddle_close']) > dataframe['bb_middleband'])
            ),
            'sell'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Exit if volatility is too high
        if last_candle['bb_width'] > 0.1:  # Bollinger Band width > 10%
            return 'high_volatility'

        # Exit if funding rate is extremely high (if you have funding rate data)
        # if abs(last_candle['funding_rate']) > 0.001:  # 0.1% funding rate
        #     return 'high_funding_rate'

        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: Current time
        :param current_rate: Current rate
        :param proposed_leverage: Proposed leverage by the strategy
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage value which is between 1.0 and max_leverage
        """
        return min(self.leverage.value, max_leverage)

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Adjust position size if necessary. Returns None if no adjustment is necessary,
        or the stake amount (in quote currency) that should be added or removed from the trade.
        This could be used to implement a dynamic position management system.
        """
        # Example: Increase position if profit is positive
        if current_profit > 0.02:  # 2% profit
            return min_stake  # Add minimum allowed stake amount

        # Example: Decrease position if loss is significant
        if current_profit < -0.05:  # 5% loss
            return -min_stake  # Remove minimum allowed stake amount

        return None  # No adjustment
