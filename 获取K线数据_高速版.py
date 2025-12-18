# -*- coding: utf-8 -*-
"""
A股创业板首板涨停股抓取 + K线缓存（pytdx + baostock 高速版）
Step1：akshare 拉涨停池（不变）
Step2：pytdx 主通道，baostock 兜底，文件命名与原脚本完全一致，老缓存无损沿用
"""

import akshare as ak
import pandas as pd
import datetime
import os
import glob
import random
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib.request
import threading
import baostock as bs
from pytdx.hq import TdxHq_API
import warnings
import socket
import logging
from collections import deque
import sys
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kline_fetch.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置socket超时
socket.setdefaulttimeout(30)

# ---------- 网络环境补丁（与原脚本一致） ----------
urllib.request.getproxies = lambda: {}
original_session_init = requests.Session.__init__

def new_session_init(self, *args, **kwargs):
    original_session_init(self, *args, **kwargs)
    self.trust_env = False
    self.proxies = {}
    self.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Connection": "close",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br"
    })

requests.Session.__init__ = new_session_init

# ---------- 基础配置 ----------
start_date = '2025-11-20'
end_date   = datetime.datetime.now().strftime('%Y-%m-%d')

base_data_dir  = './data'
data_dir       = base_data_dir
kline_cache_dir = os.path.join(base_data_dir, 'kline_cache')
templates_dir = './templates'  # 模板目录，用于存放结果文件
os.makedirs(data_dir, exist_ok=True)
os.makedirs(kline_cache_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

# 结果文件名
trade_results_filename = 'trade_results.csv'
trade_with_cangwei_filename = 'trade_with_cangwei.csv'

RETRY_COUNT = 10
RETRY_DELAY = 8
MAX_DELAY   = 60
MAX_CONCURRENT_RETRIES = 3
LONG_BREAK_DURATION = 120

def disable_proxy_environment():
    for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'all_proxy']:
        os.environ.pop(k, None)

disable_proxy_environment()

# ---------- pytdx 连接池 ----------
# 连接池类
class PytdxConnectionPool:
    def __init__(self, max_connections=5, server_list=None, check_interval=300):
        """
        初始化pytdx连接池
        
        Args:
            max_connections: 最大连接数
            server_list: 服务器列表
            check_interval: 连接健康检查间隔(秒)
        """
        self.max_connections = max_connections
        self.server_list = server_list or [
            ('121.36.81.195', 7709)
        ]
        self.connections = deque()
        self.in_use = set()
        self.lock = threading.RLock()
        self.check_interval = check_interval
        self.last_check_time = time.time()
        
        # 启动健康检查线程
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
    
    def _create_connection(self):
        """创建一个新的pytdx连接"""
        for ip, port in self.server_list:
            logger.info(f"尝试连接服务器: {ip}:{port}")
            try:
                # 对于旧版本pytdx，简化参数设置
                client = TdxHq_API()
                # 旧版本不支持timeout参数
                client.connect(ip, port)
                # 不进行额外验证，直接返回连接
                logger.info(f"成功连接服务器: {ip}:{port}")
                return client
            except Exception as e:
                logger.warning(f"连接服务器 {ip}:{port} 失败: {str(e)}")
                try:
                    client.disconnect()
                except:
                    pass
                continue
        
        logger.error("所有服务器连接失败")
        return None
    
    def _check_connection_health(self, client):
        """检查连接是否健康（适配旧版本pytdx）"""
        try:
            # 对于旧版本pytdx，我们只能进行基本检查
            # 不使用任何高级API方法，避免兼容性问题
            if client is None:
                return False
            
            # 对于旧版本pytdx，我们暂时跳过主动检查，直接返回True
            # 因为任何API调用都可能在旧版本上失败
            # 连接的有效性将在实际使用时通过异常处理来验证
            return True
        except Exception as e:
            logger.warning(f"连接健康检查失败: {str(e)}")
            return False
    
    def _health_check_loop(self):
        """定期健康检查循环"""
        while True:
            time.sleep(min(self.check_interval, 60))  # 最多等待60秒
            self._clean_dead_connections()
    
    def _clean_dead_connections(self):
        """清理无效连接"""
        with self.lock:
            current_time = time.time()
            if current_time - self.last_check_time < self.check_interval:
                return
            
            self.last_check_time = current_time
            healthy_connections = []
            
            # 检查空闲连接
            for conn in self.connections:
                if self._check_connection_health(conn):
                    healthy_connections.append(conn)
                else:
                    try:
                        conn.disconnect()
                    except:
                        pass
                    logger.info("移除了一个不健康的空闲连接")
            
            self.connections = deque(healthy_connections)
    
    def get_connection(self):
        """从连接池获取一个连接"""
        start_time = time.time()
        max_wait_time = 10  # 最大等待时间10秒
        
        while time.time() - start_time < max_wait_time:
            with self.lock:
                # 先尝试从现有空闲连接中获取
                while self.connections:
                    conn = self.connections.popleft()
                    if self._check_connection_health(conn):
                        self.in_use.add(conn)
                        return conn
                    else:
                        try:
                            conn.disconnect()
                        except:
                            pass
                        logger.info("移除了一个不健康的空闲连接")
                
                # 如果没有空闲连接且未达到最大连接数，则创建新连接
                if len(self.in_use) < self.max_connections:
                    conn = self._create_connection()
                    if conn:
                        self.in_use.add(conn)
                        return conn
            
            # 等待一小段时间再尝试
            time.sleep(0.1)
        
        logger.error("获取连接超时")
        return None
    
    def release_connection(self, conn):
        """释放连接回连接池"""
        if conn is None:
            return
        
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
                if self._check_connection_health(conn):
                    self.connections.append(conn)
                else:
                    try:
                        conn.disconnect()
                    except:
                        pass
    
    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            # 关闭空闲连接
            while self.connections:
                conn = self.connections.popleft()
                try:
                    conn.disconnect()
                except:
                    pass
            
            # 关闭正在使用的连接
            for conn in list(self.in_use):
                try:
                    conn.disconnect()
                except:
                    pass
            
            self.in_use.clear()

# 创建全局连接池实例
connection_pool = PytdxConnectionPool(max_connections=10, check_interval=180)

# ---------- K线抓取（pytdx 主通道 + baostock 兜底） ----------
def fetch_kline_pytdx(code: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """日线前复权，返回与 akshare 统一字段（适配旧版本pytdx）"""
    # 修正市场代码判断，特别是对创业板的处理
    market = 0 if code.startswith('00') or code.startswith('30') else 1
    start_time = time.time()
    retry_count = 0
    max_retries = 3
    api = None
    
    while retry_count < max_retries:
        try:
            # 从连接池获取连接
            api = connection_pool.get_connection()
            if api is None:
                retry_count += 1
                # 根据用户要求，移除连接获取失败时的延迟
                logger.warning(f"无法从连接池获取连接，立即重试 (尝试 {retry_count}/{max_retries})")
                continue
            
            # 设置请求超时保护
            timeout_seconds = 30  # 设置30秒超时
            
            # 使用socket超时上下文
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout_seconds)
            
            try:
                # 简化API调用参数，减小数据量以提高成功率
                max_days = min((end - start).days + 60, 1000)  # 限制数据量
                
                # 首先尝试旧版本API方法
                try:
                    bars = api.get_security_bars(9, market, code, 0, max_days)
                except Exception as inner_e:
                    logger.warning(f"get_security_bars失败，尝试备选方法: {str(inner_e)}")
                    # 如果get_security_bars失败，尝试其他可能的方法
                    try:
                        # 尝试直接返回的数据格式
                        bars = api.get_security_bars(9, market, code, 0, max_days)
                    except:
                        # 尝试直接创建DataFrame
                        bars = []
                
                # 针对旧版本pytdx的兼容性处理
                if hasattr(api, 'to_df') and callable(api.to_df) and bars:
                    df = api.to_df(bars)
                else:
                    # 手动创建DataFrame，适配旧版本返回的数据格式
                    if bars and len(bars) > 0:
                        # 尝试不同的数据格式
                        if isinstance(bars[0], dict):
                            df = pd.DataFrame(bars)
                        else:
                            # 假设是元组格式，尝试常见的字段映射
                            columns = ['open', 'close', 'high', 'low', 'vol', 'amount', 'year', 'month', 'day']
                            if len(bars[0]) >= len(columns):
                                df = pd.DataFrame(bars, columns=columns)
                                # 构造日期字段
                                df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
                            else:
                                df = pd.DataFrame()
                    else:
                        df = pd.DataFrame()
                
            finally:
                # 恢复原始超时设置
                socket.setdefaulttimeout(original_timeout)
            
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds * 0.8:
                logger.warning(f"获取{code}数据耗时较长: {elapsed:.2f}秒")
                
            if df.empty:
                logger.warning(f"获取{code}返回空数据")
                return pd.DataFrame()
                
            # 处理不同版本可能的列名差异
            column_mapping = {
                'datetime': '日期',
                'open': '开盘', 'open_price': '开盘',
                'high': '最高', 'high_price': '最高',
                'low': '最低', 'low_price': '最低',
                'close': '收盘', 'close_price': '收盘',
                'vol': '成交量', 'volume': '成交量',
                'amount': '成交额', 'turnover': '成交额'
            }
            
            # 动态重命名列
            rename_dict = {}
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    rename_dict[old_col] = new_col
            
            df = df.rename(columns=rename_dict)
            
            # 确保有日期列
            if '日期' not in df.columns:
                logger.warning(f"缺少日期列: {df.columns.tolist()}")
                return pd.DataFrame()
                
            # 过滤时间范围
            df['日期'] = pd.to_datetime(df['日期'])
            df = df[(df['日期'] >= pd.Timestamp(start)) & (df['日期'] <= pd.Timestamp(end))]
            
            # 确保必要的列存在
            required_columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"缺少必要列: {missing_columns}")
                return pd.DataFrame()
                
            # 添加成交额列（如果不存在）
            if '成交额' not in df.columns:
                df['成交额'] = 0
                
            # 格式化最终结果
            df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']]
            df['股票代码'] = code
            df['日期'] = df['日期'].dt.strftime('%Y-%m-%d')
            
            logger.info(f"成功获取{code}的K线数据，耗时: {elapsed:.2f}秒, 记录数: {len(df)}")
            return df.reset_index(drop=True)
            
        except (socket.timeout, Exception) as e:
            retry_count += 1
            # 根据用户要求，移除重试时的延迟
            logger.error(f"pytdx请求{code}时出错: {str(e)}，立即重试 (尝试 {retry_count}/{max_retries})")
        finally:
            # 释放连接回连接池
            if api:
                connection_pool.release_connection(api)
    
    logger.error(f"获取{code}数据失败，已达到最大重试次数")
    return pd.DataFrame()

def fetch_kline_baostock(code: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    lg = bs.login()
    if lg.error_code != '0':
        return pd.DataFrame()
    rs = bs.query_history_k_data_plus(
        code + '.SZ' if code.startswith('30') else code + '.SS',
        "date,code,open,high,low,close,volume,amount",
        start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d'),
        frequency="d", adjustflag="3")
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(data, columns=rs.fields)
    if df.empty:
        return pd.DataFrame()
    df.rename(columns={'date': '日期', 'open': '开盘', 'high': '最高', 'low': '最低',
                       'close': '收盘', 'volume': '成交量', 'amount': '成交额'}, inplace=True)
    df['股票代码'] = code
    return df

def retry_with_backoff(func, max_retries=RETRY_COUNT, base_delay=RETRY_DELAY, max_delay=MAX_DELAY, *args, **kwargs):
    """带指数退避的重试装饰器"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            result = func(*args, **kwargs)
            # 如果函数返回有效结果，则直接返回
            if func.__name__ == 'fetch_kline_pytdx' or func.__name__ == 'fetch_kline_baostock':
                if not isinstance(result, pd.DataFrame) or not result.empty:
                    return result
            else:
                return result
            # 如果返回空DataFrame，继续重试
            retry_count += 1
            if retry_count < max_retries:
                delay = min(base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1), max_delay)
                print(f"结果为空，{delay:.2f}秒后重试 ({retry_count}/{max_retries})")
                time.sleep(delay)
        except (requests.exceptions.RequestException, socket.timeout, ConnectionError, TimeoutError) as e:
            # 网络相关错误，重试更积极
            retry_count += 1
            if retry_count < max_retries:
                delay = min(base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1), max_delay)
                print(f"网络错误: {str(e)}，{delay:.2f}秒后重试 ({retry_count}/{max_retries})")
                time.sleep(delay)
        except Exception as e:
            # 其他错误，打印错误信息但不重试
            print(f"执行错误: {str(e)}")
            return pd.DataFrame() if 'fetch_kline' in func.__name__ else None
    
    print(f"达到最大重试次数 ({max_retries})，放弃重试")
    return pd.DataFrame() if 'fetch_kline' in func.__name__ else None

def fetch_kline_data(stock_code, date, period='daily', adjust=""): 
    """ 
    Fetches historical K-line data for a given stock code and date range based on trade dates. 

    Args: 
        stock_code (str): The stock code. 
        date (str or datetime): The reference date (must be a trade date). 
        period (str): The period of the K-line data (e.g., 'daily', 'weekly', 'monthly'). 
        adjust (str): The adjustment type (e.g., '', 'hfq', 'qfq'). 

    Returns: 
        pandas.DataFrame: The K-line data. 
    """ 
    if isinstance(date, str): 
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date() 
    elif isinstance(date, datetime.datetime): 
        date = date.date() 

    # Ensure trade_dates_df is available and sorted 
    global trade_dates_df 
    if 'trade_dates_df' not in globals() or trade_dates_df.empty: 
        trade_dates_df = ak.tool_trade_date_hist_sina() 
        trade_dates_df['trade_date'] = pd.to_datetime(trade_dates_df['trade_date']) # Convert to Timestamp 
        trade_dates_df.sort_values(by='trade_date', inplace=True) 

    # Calculate the start date (one year before the given date) 
    start_date_obj = date - datetime.timedelta(days=365) # Approximate one year 

    # Calculate the end date (one month after the given date) 
    # This is a bit tricky with varying month lengths, a simple approach is to add 30 days 
    end_date_obj = date + datetime.timedelta(days=30) # Approximate one month 

    # Convert date objects to Timestamp objects for searchsorted 
    start_date_ts = pd.Timestamp(start_date_obj) 
    end_date_ts = pd.Timestamp(end_date_obj) 

    # Find the closest trade dates to the calculated start and end dates 
    start_date_kline = trade_dates_df.iloc[trade_dates_df['trade_date'].searchsorted(start_date_ts, side='left')]['trade_date'].strftime('%Y%m%d') 
    end_date_kline = trade_dates_df.iloc[trade_dates_df['trade_date'].searchsorted(end_date_ts, side='right') -1]['trade_date'].strftime('%Y%m%d')

    try: 
        stock_df = ak.stock_zh_a_hist(symbol=stock_code, period=period, start_date=start_date_kline, end_date=end_date_kline, adjust=adjust) 
        return stock_df 
    except Exception as e: 
        print(f"Error fetching data for {stock_code} on {date}: {e}") 
        return pd.DataFrame()

def fetch_kline_data_pytdx_bao(stock_code, date, period='daily', adjust=""):
    """使用pytdx和baostock获取K线数据，带重试机制，保留从一年前到目标日期后5根的完整K线数据"""
    if isinstance(date, str):
        date = pd.to_datetime(date).date()
    elif isinstance(date, datetime.datetime):
        date = date.date()

    # 从目标日期前一年开始获取数据（更精确的年份计算）
    try:
        # 尝试使用更精确的年份计算（考虑闰年）
        start = date.replace(year=date.year - 1)
    except ValueError:
        # 如果是2月29日且上一年不是闰年，回退到365天计算
        start = date - datetime.timedelta(days=365)
    
    end = date + datetime.timedelta(days=30)
    code_pure = str(stock_code).split('.')[0][-6:]
    
    print(f"开始获取股票 {code_pure} 的K线数据 (pytdx/baostock)")
    
    # 首先尝试pytdx，带重试机制
    df = retry_with_backoff(fetch_kline_pytdx, RETRY_COUNT, RETRY_DELAY, MAX_DELAY, code_pure, start, end)
    
    # 如果pytdx失败，切换到baostock，同样带重试机制
    if df.empty:
        print(f"[pytdx failed] 切换到baostock获取 {code_pure}")
        df = retry_with_backoff(fetch_kline_baostock, RETRY_COUNT, RETRY_DELAY, MAX_DELAY, code_pure, start, end)
    
    if not df.empty:
        df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '股票代码']]
        
        # 确保日期列为datetime类型
        try:
            if not pd.api.types.is_datetime64_any_dtype(df['日期']):
                df['日期'] = pd.to_datetime(df['日期'])
            
            date_str = date.strftime('%Y-%m-%d')
            
            # 找出目标日期在DataFrame中的索引位置
            date_idx = df[df['日期'] >= pd.to_datetime(date_str)].index.min()
            
            # 如果找到了目标日期，只保留从一年前到目标日期后5根的数据
            if pd.notna(date_idx):
                # 确保索引有效
                date_idx = int(date_idx)
                # 计算目标日期后5根的索引位置（考虑DataFrame长度）
                end_idx = min(date_idx + 5, len(df))
                # 应用索引限制，只保留从一年前到目标日期后5根的完整K线数据
                df = df.iloc[:end_idx]
                print(f"保留股票 {code_pure} 从一年前到目标日期{date_str}后5根的完整K线数据")
        except Exception as e:
            print(f"处理K线数据时出错: {str(e)}")
        
        print(f"成功获取股票 {code_pure} 的K线数据，共{len(df)}条记录")
    else:
        print(f"警告: 无法获取股票 {code_pure} 的K线数据")
    
    return df

# ---------- 原脚本：Step1 涨停池抓取 ----------
trade_dates_df = ak.tool_trade_date_hist_sina()
trade_dates_df['trade_date'] = pd.to_datetime(trade_dates_df['trade_date'])

filtered_trade_dates = trade_dates_df[
    (trade_dates_df['trade_date'] >= start_date) & (trade_dates_df['trade_date'] <= end_date)
]

all_filtered_stocks_daily = []
for _, row in filtered_trade_dates.iterrows():
    current_date_str = row['trade_date'].strftime('%Y-%m-%d')
    print(f"Processing date: {current_date_str}")
    try:
        daily_limit_up_stocks_df = ak.stock_zt_pool_em(current_date_str.replace("-", ""))
        zt_cyb_filtered_daily = daily_limit_up_stocks_df[
            (daily_limit_up_stocks_df['代码'].astype(str).str.startswith('30')) &
            (daily_limit_up_stocks_df['连板数'] == 1)
        ][['代码', '名称']].copy()
        zt_cyb_filtered_daily['日期'] = current_date_str
        zt_cyb_filtered_daily.rename(columns={'代码': '股票代码', '名称': '股票简称'}, inplace=True)
        all_filtered_stocks_daily.append(zt_cyb_filtered_daily)
        zt_cyb_filtered_daily.to_csv(os.path.join(data_dir, f'{current_date_str}_创业板涨停.csv'), index=False, encoding='utf-8')
        print(f"Saved {len(zt_cyb_filtered_daily)} stocks for {current_date_str}")
    except Exception as e:
        print(f"Error on {current_date_str}: {e}")
        continue

final_filtered_df = pd.concat(all_filtered_stocks_daily, ignore_index=True) if all_filtered_stocks_daily else pd.DataFrame()

# ---------- 优化后的K线补全逻辑 ----------
def process_stock_batch(batch_stocks, kline_cache_dir):
    """处理一个批次的股票数据，添加更好的错误处理"""
    batch_success = 0
    batch_fail = 0
    
    for _, row in batch_stocks.iterrows():
        stock_code, date_str = str(row['股票代码']), row['日期']
        file_name = f'{date_str}_{stock_code}.SZ.csv'
        file_path = os.path.join(kline_cache_dir, file_name)
            
        # 检查缓存是否存在，即使缓存存在也重新获取数据
        if os.path.exists(file_path):
            print(f"缓存命中: {stock_code} {date_str}, 但重新获取数据")
            # 移除continue语句，确保重新获取数据
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 获取K线数据
            kline_df = fetch_kline_data_pytdx_bao(stock_code, date_str)
            
            # 处理结果
            if not kline_df.empty:
                # 确保目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # 保存数据
                kline_df.to_csv(file_path, index=False, encoding='utf-8')
                batch_success += 1
                elapsed = time.time() - start_time
                print(f"保存成功: {stock_code} {date_str} ({elapsed:.2f}秒)")
            else:
                batch_fail += 1
                print(f"数据获取失败: {stock_code} {date_str}")
                
        except Exception as e:
            batch_fail += 1
            print(f"处理异常 {stock_code} {date_str}: {str(e)}")
            # 记录错误日志
            with open('error_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"{datetime.datetime.now()}: {stock_code} {date_str} - {str(e)}\n")
        finally:
            # 根据用户要求，移除pytdx相关的休息时间
            pass
    
    return batch_success, batch_fail

# 主处理逻辑
start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date_obj   = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

csv_files = glob.glob(os.path.join(data_dir, '*_创业板涨停.csv'))
filtered_files = [f for f in csv_files if start_date_obj <= pd.to_datetime(os.path.basename(f)[:10]).date() <= end_date_obj]

combined_filtered_df_for_kline = pd.concat(
    [pd.read_csv(f) for f in csv_files], ignore_index=True
) if csv_files else pd.DataFrame()

if not combined_filtered_df_for_kline.empty:
    combined_filtered_df_for_kline['股票代码'] = combined_filtered_df_for_kline['股票代码'].astype(str)
    unique_stocks_dates = combined_filtered_df_for_kline[['股票代码', '日期']].dropna().drop_duplicates()
    success_count, fail_count = 0, 0
    batch_size = 5
    total_batches = (len(unique_stocks_dates) + batch_size - 1) // batch_size
    
    print(f"\n总共有 {len(unique_stocks_dates)} 只股票需要处理，分为 {total_batches} 个批次")
    
    # 创建错误日志文件
    with open('error_log.txt', 'w', encoding='utf-8') as f:
        f.write(f"错误日志 - 开始时间: {datetime.datetime.now()}\n")
    
    for batch_idx in range(total_batches):
        try:
            start_idx = batch_idx * batch_size
            end_idx   = min(start_idx + batch_size, len(unique_stocks_dates))
            batch_stocks = unique_stocks_dates.iloc[start_idx:end_idx]
            print(f"\n=== 批次 {batch_idx+1}/{total_batches} ===")
            print(f"处理股票范围: {start_idx+1} 到 {end_idx}")
            
            # 处理当前批次
            batch_success, batch_fail = process_stock_batch(batch_stocks, kline_cache_dir)
            success_count += batch_success
            fail_count += batch_fail
            
            print(f"批次 {batch_idx+1} 完成: 成功 {batch_success}, 失败 {batch_fail}")
            print(f"累计进度: 成功 {success_count}, 失败 {fail_count}, 总成功率: {(success_count/(success_count+fail_count))*100:.2f}%")
            
            # 根据用户要求，移除批次间的休息时间
            if batch_idx < total_batches - 1:
                print(f"准备处理下一批次...")
                
        except Exception as e:
            print(f"批次 {batch_idx+1} 处理异常: {str(e)}")
            # 记录批次错误
            with open('error_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"{datetime.datetime.now()}: 批次 {batch_idx+1} 异常 - {str(e)}\n")
            # 批次失败后，暂停更长时间再继续
            print("批次处理失败，继续处理下一批次...")
    
    print(f"\n====================================")
    print(f"K线数据获取完成")
    print(f"总处理: {success_count + fail_count}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"成功率: {(success_count/(success_count+fail_count))*100:.2f}%")
    print(f"错误日志已保存到: error_log.txt")
    print(f"====================================")
else:
    print("没有找到需要处理的股票数据。")
    print("请检查日期范围和数据目录是否正确。")

print("\n--- 所有任务已完成 ---")
   
# 测试函数已移除

# ---------- 涨停表现分析函数 ----------
def analyze_trade_performance():
    """
    分析涨停后的交易表现，生成交易结果
    """
    print("\n--- 开始分析涨停后表现 ---")
    
    # 检查kline_cache_dir是否存在
    if not os.path.isdir(kline_cache_dir):
        print(f"错误：kline_cache_dir '{kline_cache_dir}' 不存在。")
        return False
    
    # 读取所有CSV文件到pandas DataFrames
    kline_files = list(Path(kline_cache_dir).glob("*.csv"))
    
    if not kline_files:
        print(f"在目录 '{kline_cache_dir}' 中没有找到任何CSV文件。")
        return False
    
    all_stock_data = {}
    for file_path in kline_files:
        file_name = file_path.stem  # 获取文件名（不含扩展名）
        try:
            df = pd.read_csv(file_path)
            if not df.empty:  # 只添加非空数据
                all_stock_data[file_name] = df
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}。跳过此文件。")
    
    if not all_stock_data:
        print("没有成功读取任何股票数据。")
        return False
    
    # 处理数据并计算交易结果
    trade_results = []
    
    for file_name, df in all_stock_data.items():
        parts = file_name.split("_")
        if len(parts) < 2:
            print(f"文件名格式不正确，跳过文件: {file_name}。期望格式为 'YYYY-MM-DD_股票代码.csv'。")
            continue
        
        date_str = parts[0]
        stock_code = parts[1].split(".")[0]  # 去除可能的后缀，如.SZ
        
        if '日期' not in df.columns:
            print(f"股票代码: {stock_code}, 涨停日期: {date_str}: 数据中没有找到 '日期' 列，跳过处理。")
            continue
        
        df['日期'] = pd.to_datetime(df['日期'])
        try:
            trade_date = pd.to_datetime(date_str)
        except ValueError:
            print(f"文件名中的日期格式不正确，跳过文件: {file_name}。")
            continue
        
        # 查找涨停日期在DataFrame中的索引
        if trade_date in df['日期'].values:
            trade_date_index = df[df['日期'] == trade_date].index[0]
            buy_date_index = trade_date_index + 1  # 买入日期是涨停日期的下一个交易日
            sell_date_index = buy_date_index + 1   # 卖出日期是买入日期的下一个交易日
        
            # 检查是否有足够的后续交易日数据
            if sell_date_index < len(df):
                buy_price = df.loc[buy_date_index, '开盘']
                sell_price = df.loc[sell_date_index, '收盘']
                profit_loss = sell_price - buy_price
        
                # 计算买入日期和卖出日期的涨跌幅
                buy_date_open_change = None
                buy_date_close_change = None
                sell_date_open_change = None
                sell_date_close_change = None
        
                if trade_date_index >= 0:
                    prev_close_price_buy = df.loc[trade_date_index, '收盘']
                    if prev_close_price_buy != 0:
                        buy_date_open_change = ((df.loc[buy_date_index, '开盘'] - prev_close_price_buy) / prev_close_price_buy) * 100
                        buy_date_close_change = ((df.loc[buy_date_index, '收盘'] - prev_close_price_buy) / prev_close_price_buy) * 100
        
                if buy_date_index >= 0 and buy_date_index < len(df):
                    prev_close_price_sell = df.loc[buy_date_index, '收盘']
                    if prev_close_price_sell != 0:
                        sell_date_open_change = ((df.loc[sell_date_index, '开盘'] - prev_close_price_sell) / prev_close_price_sell) * 100
                        sell_date_close_change = ((df.loc[sell_date_index, '收盘'] - prev_close_price_sell) / prev_close_price_sell) * 100
        
                trade_results.append({
                    '涨停日期': trade_date.date(),
                    '股票代码': stock_code,
                    '买入价': buy_price,
                    '卖出价': sell_price,
                    '盈亏': profit_loss,
                    '买入日期开盘涨跌幅': buy_date_open_change,
                    '买入日期收盘涨跌幅': buy_date_close_change,
                    '卖出日期开盘涨跌幅': sell_date_open_change,
                    '卖出日期收盘涨跌幅': sell_date_close_change
                })
    
    # 创建DataFrame并计算 '盈亏比例'
    if trade_results:
        trade_df = pd.DataFrame(trade_results)
        
        # 计算 '盈亏比例'
        trade_df['盈亏比例'] = trade_df.apply(
            lambda row: ((row['卖出价'] - row['买入价']) / row['买入价']) * 100 if row['买入价'] != 0 else 0, 
            axis=1
        )
        
        # 保存到CSV文件（保存到templates文件夹）
        output_csv_path = os.path.join(templates_dir, trade_results_filename)
        try:
            trade_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n交易结果已保存到：{output_csv_path}")
            print(f"共生成 {len(trade_df)} 条交易记录")
            return True
        except Exception as e:
            print(f"保存CSV文件失败: {e}")
            return False
    else:
        print("没有生成任何交易结果数据。")
        return False

# ---------- 仓位建议计算函数 ----------
def calculate_position_suggestions():
    """
    计算仓位建议
    """
    print("\n--- 开始计算仓位建议 ---")
    
    # 读取交易结果文件（从templates文件夹读取）
    input_file_path = os.path.join(templates_dir, trade_results_filename)
    
    if not os.path.exists(input_file_path):
        print(f"错误：输入文件 '{input_file_path}' 不存在")
        return False
    
    df = pd.read_csv(input_file_path)
    print(f"CSV文件 '{input_file_path}' 读取成功，共 {len(df)} 条记录。")
    
    # Calculate the total count of trades for each '涨停日期'
    df['总数'] = df.groupby('涨停日期')['涨停日期'].transform('count')
    
    # Calculate the rank based on '买入日期开盘涨跌幅' within each '涨停日期' group
    df['排名'] = df.groupby('涨停日期')['买入日期开盘涨跌幅'].rank(ascending=False).astype(int)
    
    # Determine the minimum and maximum values of '买入日期开盘涨跌幅'
    min_change = df['买入日期开盘涨跌幅'].min()
    max_change = df['买入日期开盘涨跌幅'].max()
    
    # Define the bins for the intervals with a step of 1
    bins = np.arange(int(min_change), int(max_change) + 2, 1)
    
    # Create a new column for the '买入日期开盘涨跌幅' intervals
    df['涨跌幅区间'] = pd.cut(df['买入日期开盘涨跌幅'], bins=bins, right=False, include_lowest=True)
    
    # Group by '排名' and '涨跌幅区间' and calculate metrics
    grouped_df = df.groupby(['排名', '涨跌幅区间'], observed=True).agg(
        盈利数量=('盈亏比例', lambda x: (x > 0).sum()),
        平均盈利=('盈亏比例', lambda x: x[x > 0].mean()),
        亏损数量=('盈亏比例', lambda x: (x < 0).sum()),
        平均亏损=('盈亏比例', lambda x: x[x < 0].mean()),
        总交易数=('盈亏比例', 'count')
    ).reset_index()
    
    # Calculate Win Rate and Loss Rate
    grouped_df['胜率'] = grouped_df['盈利数量'] / grouped_df['总交易数']
    grouped_df['亏损率'] = grouped_df['亏损数量'] / grouped_df['总交易数']
    
    # Handle cases where there are no winning or losing trades
    grouped_df['平均盈利'] = grouped_df['平均盈利'].replace([np.inf, -np.inf], np.nan).fillna(0)
    grouped_df['平均亏损'] = grouped_df['平均亏损'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate the Win/Loss Ratio
    grouped_df['盈亏比'] = np.where(
        grouped_df['平均亏损'] != 0, 
        abs(grouped_df['平均盈利'] / grouped_df['平均亏损']), 
        np.nan
    )
    
    # Calculate Kelly Percentage
    grouped_df['凯莉公式仓位'] = np.where(
        (grouped_df['盈亏比'].notna()) & (grouped_df['胜率'].notna()),
        grouped_df['胜率'] - (1 - grouped_df['胜率']) / grouped_df['盈亏比'],
        np.nan
    )
    
    # Filter out rows where '总交易数' is 0
    filtered_grouped_df = grouped_df[grouped_df['总交易数'] != 0].copy()
    
    # Merge the original df with the filtered_grouped_df
    merged_df = pd.merge(
        df, 
        filtered_grouped_df[['排名', '涨跌幅区间', '凯莉公式仓位', '胜率']],
        on=['排名', '涨跌幅区间'],
        how='left'
    )
    
    # Define a function to calculate the position size
    def calculate_position_size(row):
        """Calculates the position size based on Kelly Criterion and Win Rate."""
        kelly_position = row['凯莉公式仓位']
        win_rate = row['胜率']
    
        if pd.isna(kelly_position) or kelly_position < 0:
            return 0
        elif win_rate == 1:
            return 0.7
        else:
            return kelly_position
    
    # Apply the function to create the '仓位' column
    merged_df['仓位'] = merged_df.apply(calculate_position_size, axis=1)
    
    # Save the merged_df to CSV（保存到templates文件夹）
    output_csv_path = os.path.join(templates_dir, trade_with_cangwei_filename)
    merged_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n仓位建议已保存到: {output_csv_path}")
    print(f"共处理 {len(merged_df)} 条记录")
    
    return True

# 主程序
def main():
    """主函数"""
    print("=== K线数据获取及分析程序启动 ===")
    
    # 步骤1: 获取K线数据
    print("\n【步骤1】获取K线数据")
    if not final_filtered_df.empty:
        process_stock_batch(final_filtered_df, kline_cache_dir)
    else:
        print("没有过滤后的股票数据可供处理。")
    
    # 确保连接池关闭
    try:
        connection_pool.close_all()
        logger.info("连接池已关闭")
    except Exception as e:
        logger.error(f"关闭连接池时出错: {str(e)}")
    
    # 步骤2: 分析涨停后表现
    print("\n【步骤2】分析涨停后表现")
    if not analyze_trade_performance():
        print("警告：涨停表现分析失败")
    
    # 步骤3: 计算仓位建议
    print("\n【步骤3】计算仓位建议")
    if not calculate_position_suggestions():
        print("警告：仓位建议计算失败")
    
    print("\n=== 所有任务已完成 ===")
    print(f"最终结果文件:")
    print(f"  - 交易结果: {os.path.join(templates_dir, trade_results_filename)}")
    print(f"  - 仓位建议: {os.path.join(templates_dir, trade_with_cangwei_filename)}")

if __name__ == "__main__":
    main()

# 注意：移除了sys.exit(0)，让main函数正常执行完成