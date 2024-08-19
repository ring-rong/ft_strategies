import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import logging

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, stoploss_from_open, RealParameter
from pandas import DataFrame, Series
from datetime import datetime, timezone
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


class ClucHAnix_hhll_Futures(IStrategy):
    """
    Futures strategy based on the ClucHAnix_hhll strategy.
    """
    can_short: bool = True
    #hypered params
    buy_params = {
        ##
        "max_slip": 0.73,
        ##
        "bbdelta_close": 0.01846,
        "bbdelta_tail": 0.98973,
        "close_bblower": 0.00785,
        "closedelta_close": 0.01009,
        "rocr_1h": 0.5411,
        ##
        "buy_hh_diff_48": 6.867,
        "buy_ll_diff_48": -12.884,
    }

    # Sell hyperspace params:
    sell_params = {
        "pPF_1": 0.011,
        "pPF_2": 0.064,
        "pSL_1": 0.011,
        "pSL_2": 0.062,

        # sell signal params
        "high_offset": 0.907,
        "high_offset_2": 1.211,
        "sell_bbmiddle_close": 0.97286,
        "sell_fisher": 0.48492,
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

    # Futures stoploss:
    stoploss = -0.99  # use custom stoploss

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = False

    leverage = 10.0 # 10x leverage

    # Timeframe and startup candle count
    timeframe = '15m'
    startup_candle_count = 168

    process_only_new_candles = True
    # order_types = {
    #     'buy': 'market',
    #     'sell': 'market',
    #     'stoploss': 'market',
    #     'stoploss_on_exchange': True,
    # }

    # buy params
    is_optimize_clucHA = False
    rocr_1h = RealParameter(0.5, 1.0, default=0.54904, space='buy', optimize = is_optimize_clucHA )
    bbdelta_close = RealParameter(0.0005, 0.02, default=0.01965, space='buy', optimize = is_optimize_clucHA )
    closedelta_close = RealParameter(0.0005, 0.02, default=0.00556, space='buy', optimize = is_optimize_clucHA )
    bbdelta_tail = RealParameter(0.7, 1.0, default=0.95089, space='buy', optimize = is_optimize_clucHA )
    close_bblower = RealParameter(0.0005, 0.02, default=0.00799, space='buy', optimize = is_optimize_clucHA )

    is_optimize_hh_ll = False
    buy_hh_diff_48 = DecimalParameter(0.0, 15, default=1.087 , optimize = is_optimize_hh_ll )
    buy_ll_diff_48 = DecimalParameter(-23, 40, default=1.087 , optimize = is_optimize_hh_ll )

    ## Slippage params
    is_optimize_slip = False
    max_slip = DecimalParameter(0.33, 0.80, default=0.33, decimals=3, optimize=is_optimize_slip , space='buy', load=True)

    # sell params
    is_optimize_sell = False
    sell_fisher = RealParameter(0.1, 0.5, default=0.38414, space='sell', optimize = is_optimize_sell)
    sell_bbmiddle_close = RealParameter(0.97, 1.1, default=1.07634, space='sell', optimize = is_optimize_sell)
    high_offset          = DecimalParameter(0.90, 1.2, default=sell_params['high_offset'], space='sell', optimize = is_optimize_sell)
    high_offset_2        = DecimalParameter(0.90, 1.5, default=sell_params['high_offset_2'], space='sell', optimize = is_optimize_sell)

    is_optimize_trailing = False
    pPF_1 = DecimalParameter(0.011, 0.020, default=0.016, decimals=3, space='sell', load=True, optimize = is_optimize_trailing)
    pSL_1 = DecimalParameter(0.011, 0.020, default=0.011, decimals=3, space='sell', load=True, optimize = is_optimize_trailing)
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True, optimize = is_optimize_trailing)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True, optimize = is_optimize_trailing)

    def leverage(self, pair: str, current_time: datetime) -> float:
        """
        Return the leverage to be used for a specific pair.
        """
        return self.leverage  # Use the leverage defined in the strategy
    
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        # hard stoploss profit
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value

        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        sl_profit = -0.99

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = -0.99

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99
    
        return stoploss_from_open(sl_profit, current_profit)
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Set Up Bollinger Bands
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        # BB 20
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['ema_24'] = ta.EMA(dataframe['close'], timeperiod=24)
        dataframe['ema_200'] = ta.EMA(dataframe['close'], timeperiod=200)

        # SMA
        dataframe['sma_9'] = ta.SMA(dataframe['close'], timeperiod=9)
        dataframe['sma_200'] = ta.SMA(dataframe['close'], timeperiod=200)

        # HMA
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        # volume
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        # ROCR
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        # hh48
        dataframe['hh_48'] = ta.MAX(dataframe['high'], 48)
        dataframe['hh_48_diff'] = (dataframe['hh_48'] - dataframe['close']) / dataframe['hh_48'] * 100

        # ll48
        dataframe['ll_48'] = ta.MIN(dataframe['low'], 48)
        dataframe['ll_48_diff'] = (dataframe['close'] - dataframe['ll_48']) / dataframe['ll_48'] * 100

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # RSI
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # sma dec 20
        dataframe['sma_200_dec_20'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)

        # EMA of VWMA Oscillator
        dataframe['ema_vwma_osc_32'] = ema_vwma_osc(dataframe, 32)
        dataframe['ema_vwma_osc_64'] = ema_vwma_osc(dataframe, 64)
        dataframe['ema_vwma_osc_96'] = ema_vwma_osc(dataframe, 96)

        # CMF
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # 1h tf
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)

        informative['sma_200'] = ta.SMA(informative['close'], timeperiod=200)

        informative['hl_pct_change_48'] = range_percent_change(informative, 'HL', 48)
        informative['hl_pct_change_36'] = range_percent_change(informative, 'HL', 36)
        informative['hl_pct_change_24'] = range_percent_change(informative, 'HL', 24)
        informative['sma_200_dec_20'] = informative['sma_200'] < informative['sma_200'].shift(20)

        # CMF
        informative['cmf'] = chaikin_money_flow(informative, 20)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rocr_1h'].gt(self.rocr_1h.value))
                &
                (
                    (
                        (dataframe['lower'].shift().gt(0))
                        &
                        (dataframe['bbdelta'].gt(dataframe['ha_close'] * self.bbdelta_close.value))
                        &
                        (dataframe['closedelta'].gt(dataframe['ha_close'] * self.closedelta_close.value))
                        &
                        (dataframe['tail'].lt(dataframe['bbdelta'] * self.bbdelta_tail.value))
                        &
                        (dataframe['ha_close'].lt(dataframe['lower'].shift()))
                        &
                        (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
                    )
                    |
                    (
                        (dataframe['ha_close'] < dataframe['ema_slow'])
                        &
                        (dataframe['ha_close'] < self.close_bblower.value * dataframe['bb_lowerband'])
                    )
                )
                &
                (dataframe['hh_48_diff'] > self.buy_hh_diff_48.value)
                &
                (dataframe['ll_48_diff'] > self.buy_ll_diff_48.value)
            ),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (dataframe['rocr_1h'].lt(self.rocr_1h.value))  # Inverse of the buy condition
            &
            (
                (
                    (dataframe['upper'].shift().lt(0))  # Inverse of lower > 0
                    &
                    (dataframe['bbdelta'].lt(dataframe['ha_close'] * self.bbdelta_close.value))  # Inverse of bbdelta > ha_close * threshold
                    &
                    (dataframe['closedelta'].lt(dataframe['ha_close'] * self.closedelta_close.value))  # Inverse of closedelta > ha_close * threshold
                    &
                    (dataframe['tail'].gt(dataframe['bbdelta'] * self.bbdelta_tail.value))  # Inverse of tail < bbdelta * threshold
                    &
                    (dataframe['ha_close'].gt(dataframe['upper'].shift()))  # Inverse of ha_close < lower.shift()
                    &
                    (dataframe['ha_close'].ge(dataframe['ha_close'].shift()))  # Inverse of ha_close <= ha_close.shift()
                )
                |
                (
                    (dataframe['ha_close'] > dataframe['ema_slow'])  # Inverse of ha_close < ema_slow
                    &
                    (dataframe['ha_close'] > self.close_bbupper.value * dataframe['bb_upperband'])  # Inverse of ha_close < close_bblower * bb_lowerband
                )
            )
            &
            (dataframe['hh_48_diff'] < self.sell_hh_diff_48.value)  # Inverse of hh_48_diff > threshold
            &
            (dataframe['ll_48_diff'] < self.sell_ll_diff_48.value)  # Inverse of ll_48_diff > threshold
        ),
        'sell'
    ] = 1
    return dataframe

    def populate_short_entry(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['fisher'] > self.sell_fisher.value)
                    &
                    (dataframe['ha_high'].le(dataframe['ha_high'].shift(1)))
                    &
                    (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2)))
                    &
                    (dataframe['ha_close'].le(dataframe['ha_close'].shift(1)))
                    &
                    (dataframe['ema_fast'] > dataframe['ha_close'])
                    &
                    ((dataframe['ha_close'] * self.sell_bbmiddle_close.value) > dataframe['bb_middleband'])
                )
                |
                (
                    (dataframe['close'] > (dataframe['ema_fast'] * self.high_offset_2.value))  # Access .value
                    &
                    (dataframe['close'].shift(1) > (dataframe['ema_fast'] * self.high_offset.value))  # Access .value
                )
            ),
            'sell'
        ] = 1
        return dataframe

    def populate_exit_short(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (
                (dataframe['fisher'] < self.exit_fisher.value)  # Inverse of the sell condition
                &
                (dataframe['ha_low'].ge(dataframe['ha_low'].shift(1)))  # Inverse of ha_high decreasing
                &
                (dataframe['ha_low'].shift(1).ge(dataframe['ha_low'].shift(2)))  # Same inverse logic applied
                &
                (dataframe['ha_close'].ge(dataframe['ha_close'].shift(1)))  # Inverse of ha_close decreasing
                &
                (dataframe['ema_fast'] < dataframe['ha_close'])  # Inverse of ema_fast > ha_close
                &
                ((dataframe['ha_close'] * self.exit_bbmiddle_close.value) < dataframe['bb_middleband'])  # Inverse logic applied
            )
            |
            (
                (dataframe['close'] < (dataframe['ema_fast'] * self.low_offset_2.value))  # Inverse of the sell condition
                &
                (dataframe['close'].shift(1) < (dataframe['ema_fast'] * self.low_offset.value))  # Inverse of the sell condition
            )
        ),
        'exit_short'
    ] = 1
    return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.populate_buy_trend(dataframe, metadata)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.populate_sell_trend(dataframe, metadata)
    
        # Volume Weighted Moving Average
def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    vwma = vwma.fillna(0, inplace=True)
    return vwma

# Exponential moving average of a volume weighted simple moving average
def ema_vwma_osc(dataframe, len_slow_ma):
    slow_ema = Series(ta.EMA(vwma(dataframe, len_slow_ma), len_slow_ma))
    return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100

def range_percent_change(dataframe: DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        if method == 'HL':
            return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe['low'].rolling(length).min()
        elif method == 'OC':
            return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe['close'].rolling(length).min()
        else:
            raise ValueError(f"Method {method} not defined!")

# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')
class ClucHAnix_hhll_TB_Futures_LongShort(ClucHAnix_hhll_Futures):

    process_only_new_candles = True

    custom_info_trail_buy = dict()
    custom_info_trail_sell = dict()

    # Trailing buy parameters (Long)
    trailing_buy_order_enabled = True
    trailing_expire_seconds = 1800

    trailing_buy_uptrend_enabled = False
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.02
    trailing_buy_max_buy = 0.000

    # Trailing sell parameters (Short)
    trailing_sell_order_enabled = True
    trailing_sell_expire_seconds = 1800

    trailing_sell_downtrend_enabled = False
    trailing_sell_expire_seconds_downtrend = 90
    min_downtrend_trailing_profit = 0.02

    trailing_sell_max_stop = 0.02
    trailing_sell_max_sell = 0.000

    init_trailing_dict = {
        'trailing_order_started': False,
        'trailing_order_uplimit': 0,
        'start_trailing_price': 0,
        'trade_tag': None,
        'start_trailing_time': None,
        'offset': 0,
        'allow_trailing': False,
    }

    def trailing_order(self, pair, reinit=False, order_type='buy'):
        custom_info = self.custom_info_trail_buy if order_type == 'buy' else self.custom_info_trail_sell
        if not pair in custom_info:
            custom_info[pair] = dict()
        if reinit or not f'trailing_{order_type}' in custom_info[pair]:
            custom_info[pair][f'trailing_{order_type}'] = self.init_trailing_dict.copy()
        return custom_info[pair][f'trailing_{order_type}']

    def trailing_order_info(self, pair: str, current_price: float, order_type='buy'):
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_order = self.trailing_order(pair, order_type=order_type)

        duration = 0
        try:
            duration = (current_time - trailing_order['start_trailing_time'])
        except TypeError:
            duration = 0
        finally:
            logger.info(
                f"pair: {pair} : "
                f"start: {trailing_order['start_trailing_price']:.4f}, "
                f"duration: {duration}, "
                f"current: {current_price:.4f}, "
                f"uplimit: {trailing_order['trailing_order_uplimit']:.4f}, "
                f"profit: {self.current_trailing_profit_ratio(pair, current_price, order_type)*100:.2f}%, "
                f"offset: {trailing_order['offset']}"
            )

    def current_trailing_profit_ratio(self, pair: str, current_price: float, order_type='buy') -> float:
        trailing_order = self.trailing_order(pair, order_type=order_type)
        if trailing_order['trailing_order_started']:
            return (trailing_order['start_trailing_price'] - current_price) / trailing_order['start_trailing_price'] if order_type == 'buy' else (current_price - trailing_order['start_trailing_price']) / trailing_order['start_trailing_price']
        else:
            return 0

    def trailing_order_offset(self, dataframe, pair: str, current_price: float, order_type='buy'):
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price, order_type)
        default_offset = 0.005

        trailing_order = self.trailing_order(pair, order_type=order_type)
        if not trailing_order['trailing_order_started']:
            return default_offset

        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_order['start_trailing_time']
        if trailing_duration.total_seconds() > (self.trailing_expire_seconds if order_type == 'buy' else self.trailing_sell_expire_seconds):
            if ((current_trailing_profit_ratio > 0 and order_type == 'buy') or (current_trailing_profit_ratio < 0 and order_type == 'sell')) and (last_candle[f'{order_type}'] == 1):
                return 'force' + order_type
            else:
                return None
        elif (self.trailing_buy_uptrend_enabled if order_type == 'buy' else self.trailing_sell_downtrend_enabled) and (trailing_duration.total_seconds() < (self.trailing_expire_seconds_uptrend if order_type == 'buy' else self.trailing_sell_expire_seconds_downtrend)) and (current_trailing_profit_ratio < (-1 * (self.min_uptrend_trailing_profit if order_type == 'buy' else self.min_downtrend_trailing_profit))):
            return 'force' + order_type

        if (order_type == 'buy' and current_trailing_profit_ratio < 0) or (order_type == 'sell' and current_trailing_profit_ratio > 0):
            return default_offset

        trailing_offset_map = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_offset_map:
            if current_trailing_profit_ratio > key if order_type == 'buy' else current_trailing_profit_ratio < key:
                return trailing_offset_map[key]

        return default_offset

    # Indicator population should account for both long and short trends
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_order(metadata['pair'], order_type='buy')
        self.trailing_order(metadata['pair'], order_type='sell')
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)

        if val:
            if order_type == 'buy' and self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                val = self.handle_trailing_entry(pair, rate, dataframe, 'buy')

            elif order_type == 'sell' and self.trailing_sell_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                val = self.handle_trailing_entry(pair, rate, dataframe, 'sell')

        return val

    def handle_trailing_entry(self, pair, rate, dataframe, order_type):
        val = False
        trailing_order = self.trailing_order(pair, order_type=order_type)
        trailing_offset = self.trailing_order_offset(dataframe, pair, rate, order_type)

        if trailing_order['allow_trailing']:
            if not trailing_order['trailing_order_started'] and dataframe.iloc[-1].squeeze()[order_type] == 1:
                trailing_order['trailing_order_started'] = True
                trailing_order['trailing_order_uplimit'] = dataframe.iloc[-1].squeeze()['close']
                trailing_order['start_trailing_price'] = dataframe.iloc[-1].squeeze()['close']
                trailing_order['trade_tag'] = dataframe.iloc[-1].squeeze().get(f'{order_type}_tag', f'{order_type} signal')
                trailing_order['start_trailing_time'] = datetime.now(timezone.utc)
                trailing_order['offset'] = 0

                self.trailing_order_info(pair, rate, order_type)
                logger.info(f'Start trailing {order_type} for {pair} at {trailing_order["start_trailing_price"]}')

            elif trailing_order['trailing_order_started']:
                if trailing_offset == f'force{order_type}':
                    val = True
                    self.trailing_order_info(pair, rate, order_type)
                    logger.info(f"Price OK for {pair}, {order_type} order may be triggered if margin is sufficient")

                elif trailing_offset is None:
                    self.trailing_order(pair, reinit=True, order_type=order_type)
                    logger.info(f'STOP trailing {order_type} for {pair} because "trailing order offset" returned None')

                elif rate < trailing_order['trailing_order_uplimit'] if order_type == 'buy' else rate > trailing_order['trailing_order_uplimit']:
                    old_uplimit = trailing_order["trailing_order_uplimit"]
                    self.custom_info_trail_buy[pair][f'trailing_{order_type}']['trailing_order_uplimit'] = min(rate * (1 + trailing_offset), trailing_order['trailing_order_uplimit'])
                    trailing_order['offset'] = trailing_offset
                    self.trailing_order_info(pair, rate, order_type)
                    logger.info(f'Update trailing {order_type} for {pair}: {old_uplimit} -> {trailing_order["trailing_order_uplimit"]}')

                elif rate < trailing_order['start_trailing_price'] * (1 + self.trailing_buy_max_buy if order_type == 'buy' else self.trailing_sell_max_sell):
                    val = True
                    self.trailing_order_info(pair, rate, order_type)
                    logger.info(f"Current price ({rate}) > uplimit ({trailing_order['trailing_order_uplimit']}) and lower than starting price. {order_type.capitalize()} order for {pair} may be triggered")

                elif rate > trailing_order['start_trailing_price'] * (1 + self.trailing_buy_max_stop if order_type == 'buy' else self.trailing_sell_max_stop):
                    self.trailing_order(pair, reinit=True, order_type=order_type)
                    self.trailing_order_info(pair, rate, order_type)
                    logger.info(f'STOP trailing {order_type} for {pair} because price exceeded stop limit')
                else:
                    self.trailing_order_info(pair, rate, order_type)
                    logger.info(f'Price too high for {pair}!')

        return val

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)
        self.handle_trailing_entry(metadata['pair'], dataframe.iloc[-1].squeeze()['close'], dataframe, 'buy')
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_sell_trend(dataframe, metadata)
        self.handle_trailing_entry(metadata['pair'], dataframe.iloc[-1].squeeze()['close'], dataframe, 'sell')
        return dataframe
