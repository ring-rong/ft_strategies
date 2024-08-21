import pandas as pd
import numpy as np
import talib.abstract as ta
import logging

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
from statistics import mean

from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy.interface import IStrategy

class RingRong5m(IStrategy):
    can_short: bool = True
    timeframe = '5m'
    stoploss = -0.05
    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pd.set_option('display.max_rows', 100000)
        pd.set_option('display.precision', 10)
        pd.set_option('mode.chained_assignment', None)

        df['rsi_7'] = ta.RSI(df['close'], timeperiod=7).round(2)
        df['min_level'] = 0
        df['max_level'] = 0
        df['buy_short'] = 0
        df['buy_long'] = 0
        df['buy_short2'] = 0
        df['buy_long2'] = 0
        df['i_close'] = 0
        df['i_open'] = 0
        df['i_low'] = 0
        df['i_high'] = 0

        logging.getLogger('freqtrade').info(str(metadata))

        df = self.do_heikin_ashi(df)
        df = self.do_long(df)
        df = self.do_short(df)

        return df

    def do_long(self, df: pd.DataFrame) -> pd.DataFrame:
        n = 200
        df['min_local'] = df.iloc[signal.argrelextrema(df.c.values, np.less_equal, order=n)[0]]['c']
        min = df['c'].max()

        times = df.query(f'min_local > 0')
        prices = []

        if len(times) > 0:
            for i in range(500, len(df)):
                if df.loc[i, 'min_local'] > 0:
                    close = df.loc[i, 'c']

                    chunk = df[:i - 10]
                    chunk['min_local'] = chunk.iloc[signal.argrelextrema(chunk.c.values, np.less_equal, order=n)[0]]['c']

                    time_chunk = chunk.query(f'min_local > 0')

                    for x, row in time_chunk.iterrows():
                        close_chunk = time_chunk.loc[x, 'c']
                        prices.append(close_chunk)

                    for p in prices:
                        diff = self.diff_percentage(p, close)

                        if 0 < diff < 0.3:
                            logging.getLogger('freqtrade').info(str([i, '+++', diff, p]))
                            df.loc[i, 'min_level'] = 1

                for x in range(i - 2, i):
                    if df.loc[x, 'min_level'] > 0:
                        close = df.loc[i, 'c']

                        df_tail = df[i - 200: i].query(f'c < {close}')

                        if len(df_tail) > 0:
                            logging.getLogger('freqtrade').info(
                                str([i, '++++', min, diff, '---', prices, 'long ++++++'])
                            )
                            df.loc[i, 'buy_long'] = 1
                            df.loc[i, 'i_low'] = df.loc[x, 'low'].astype(df['i_low'].dtype)
                            df.loc[i, 'i_high'] = df.loc[x, 'high'].astype(df['i_high'].dtype)
                            df.loc[i, 'i_open'] = df.loc[x, 'o'].astype(df['i_open'].dtype)
                            df.loc[i, 'i_close'] = df.loc[x, 'c'].astype(df['i_close'].dtype)

                for x in range(i - 100, i):
                    if df.loc[x, 'min_level'] > 0:
                        diff = self.diff_percentage(df.loc[x, 'c'], df.loc[i, 'c'])
                        if 0 < diff < 1:
                            logging.getLogger('freqtrade').info(
                                str([i, '---- long2 ----'])
                            )
                            df.loc[i, 'buy_long2'] = 1
                            df.loc[i, 'i_low'] = df.loc[x, 'low'].astype(df['i_low'].dtype)
                            df.loc[i, 'i_high'] = df.loc[x, 'high'].astype(df['i_high'].dtype)
                            df.loc[i, 'i_open'] = df.loc[x, 'o'].astype(df['i_open'].dtype)
                            df.loc[i, 'i_close'] = df.loc[x, 'c'].astype(df['i_close'].dtype)

        return df

    def do_short(self, df: pd.DataFrame) -> pd.DataFrame:
        n = 200
        df['max_local'] = df.iloc[signal.argrelextrema(df.c.values, np.greater_equal, order=n)[0]]['c']
        max = df['c'].min()

        times = df.query(f'max_local > 0')
        prices = []

        if len(times) > 0:
            for i in range(500, len(df)):
                if df.loc[i, 'max_local'] > 0:
                    close = df.loc[i, 'c']

                    chunk = df[:i - 10]
                    chunk['max_x'] = chunk.iloc[signal.argrelextrema(chunk.c.values, np.greater_equal, order=n)[0]]['c']

                    time_chunk = chunk.query(f'max_x > 0')

                    for x, row in time_chunk.iterrows():
                        close_chunk = time_chunk.loc[x, 'c']
                        prices.append(close_chunk)

                    for p in prices:
                        diff = self.diff_percentage(p, close)

                        if 0 < diff < 0.3:
                            logging.getLogger('freqtrade').info(str([i, '---', diff, p]))
                            df.loc[i, 'max_level'] = 1

                for x in range(i - 2, i):
                    if df.loc[x, 'max_level'] > 0:
                        close = df.loc[i, 'c']

                        df_tail = df[i - 200: i].query(f'c > {close}')

                        if len(df_tail) > 0:
                            logging.getLogger('freqtrade').info(
                                str([i, '---', max, diff, '---', prices, 'short -----'])
                            )
                            df.loc[i, 'buy_short'] = 1
                            df.loc[i, 'i_low'] = df.loc[x, 'low'].astype(df['i_low'].dtype)
                            df.loc[i, 'i_high'] = df.loc[x, 'high'].astype(df['i_high'].dtype)
                            df.loc[i, 'i_open'] = df.loc[x, 'o'].astype(df['i_open'].dtype)
                            df.loc[i, 'i_close'] = df.loc[x, 'c'].astype(df['i_close'].dtype)

                for x in range(i - 100, i):
                    if df.loc[x, 'max_level'] > 0:
                        diff = self.diff_percentage(df.loc[x, 'c'], df.loc[i, 'c'])
                        if 0 < diff < 1:
                            logging.getLogger('freqtrade').info(
                                str([i, '---- short2 ----'])
                            )
                            df.loc[i, 'buy_short2'] = 1
                            df.loc[i, 'i_low'] = df.loc[x, 'low'].astype(df['i_low'].dtype)
                            df.loc[i, 'i_high'] = df.loc[x, 'high'].astype(df['i_high'].dtype)
                            df.loc[i, 'i_open'] = df.loc[x, 'o'].astype(df['i_open'].dtype)
                            df.loc[i, 'i_close'] = df.loc[x, 'c'].astype(df['i_close'].dtype)

        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df.loc[(df['buy_short'] > 0), 'enter_short'] = 1
        df.loc[(df['buy_short2'] > 0), 'enter_short'] = 1

        df.loc[(df['buy_long'] > 0), 'enter_long'] = 1
        df.loc[(df['buy_long2'] > 0), 'enter_long'] = 1

        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df.loc[
            (df['rsi_7'] < 10),
            'exit_short'
        ] = 1
        df.loc[
            (df['rsi_7'] > 90),
            'exit_long'
        ] = 1

        return df

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:

        return 10

    def diff_percentage(self, v2, v1) -> float:
        diff = ((v2 - v1) / ((v2 + v1) / 2)) * 100
        diff = np.round(diff, 4)

        return np.abs(diff)

    def do_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        df['h'] = df.apply(lambda x: max(x['high'], x['open'], x['close']), axis=1)
        df['l'] = df.apply(lambda x: min(x['low'], x['open'], x['close']), axis=1)

        for i in range(1, len(df)):
            df.loc[i, 'c'] = 1 / 4 * (
                df['open'].iloc[i] +
                df['close'].iloc[i] +
                df['high'].iloc[i] +
                df['low'].iloc[i]
            )
            df.loc[i, 'o'] = 1 / 2 * (
                df['open'].iloc[i - 1] +
                df['close'].iloc[i - 1]
            )

        return df

    def confirm_trade_entry(self,
                            pair: str,
                            order_type: str,
                            amount: float,
                            rate: float,
                            time_in_force: str,
                            current_time: datetime,
                            entry_tag: Optional[str],
                            side: str,
                            **kwargs) -> bool:
        df, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        print('------------------- confirm_trade_entry -------------', pair, rate)

        mean = (
            df['i_open'].iat[-1] +
            df['i_close'].iat[-1]
        ) / 2

        if mean == 0:
            return False

        if side == 'long':
            return rate < mean

        if side == 'short':
            return rate > mean

        return False
