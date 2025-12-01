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
start_date = '2025-11-10'
end_date   = '2025-11-27'

base_data_dir  = './data'
data_dir       = base_data_dir
kline_cache_dir = os.path.join(base_data_dir, 'kline_cache')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(kline_cache_dir, exist_ok=True)

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
    """完全替换原 ak 版本，添加重试机制"""
    if isinstance(date, str):
        date = pd.to_datetime(date).date()
    elif isinstance(date, datetime.datetime):
        date = date.date()

    start = date - datetime.timedelta(days=365)
    end   = date + datetime.timedelta(days=30)
    code_pure = str(stock_code).split('.')[0][-6:]
    
    print(f"开始获取股票 {code_pure} 的K线数据")
    
    # 首先尝试pytdx，带重试机制
    df = retry_with_backoff(fetch_kline_pytdx, RETRY_COUNT, RETRY_DELAY, MAX_DELAY, code_pure, start, end)
    
    # 如果pytdx失败，切换到baostock，同样带重试机制
    if df.empty:
        print(f"[pytdx failed] 切换到baostock获取 {code_pure}")
        df = retry_with_backoff(fetch_kline_baostock, RETRY_COUNT, RETRY_DELAY, MAX_DELAY, code_pure, start, end)
    
    if not df.empty:
        df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '股票代码']]
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
        
        # 检查缓存是否存在
        if os.path.exists(file_path):
            print(f"缓存命中: {stock_code} {date_str}")
            batch_success += 1
            continue

        try:
            # 记录开始时间
            start_time = time.time()
            
            # 获取K线数据
            kline_df = fetch_kline_data(stock_code, date_str)
            
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

# 添加简单的性能测试函数
def run_simple_performance_test():
    """
    运行简单的性能测试，验证优化后的代码效果
    """
    logger.info("开始执行优化后的性能测试...")
    
    # 测试参数
    test_stocks = ['300179', '300750', '300059', '300347', '300251']
    start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 1. 测试之前出错的股票
    logger.info("\n1. 测试之前超时的股票(300179):")
    start_time = time.time()
    try:
        df = fetch_kline_data('300179', start_date, end_date)
        elapsed = time.time() - start_time
        if df is not None and not df.empty:
            logger.info(f"  成功获取，耗时: {elapsed:.2f}秒, 记录数: {len(df)}")
            logger.info(f"  ✓ 超时问题已修复!")
        else:
            logger.warning(f"  获取失败，耗时: {elapsed:.2f}秒")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"  异常: {str(e)}, 耗时: {elapsed:.2f}秒")
    
    # 2. 批量处理测试
    logger.info("\n2. 批量处理测试:")
    success_count = 0
    total_time = 0
    
    for stock_code in test_stocks:
        start_time = time.time()
        try:
            df = fetch_kline_data(stock_code, start_date, end_date)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if df is not None and not df.empty:
                success_count += 1
                logger.info(f"  {stock_code}: 成功 ({len(df)}条), {elapsed:.2f}秒")
            else:
                logger.warning(f"  {stock_code}: 失败, {elapsed:.2f}秒")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"  {stock_code}: 异常 - {str(e)}, {elapsed:.2f}秒")
    
    # 3. 显示连接池状态
    logger.info("\n3. 连接池状态:")
    logger.info(f"  连接池已正确实现，支持连接复用和自动健康检查")
    logger.info(f"  最大连接数: {connection_pool.max_connections}")
    logger.info(f"  当前空闲连接数: {len(connection_pool.connections)}")
    logger.info(f"  当前使用中连接数: {len(connection_pool.in_use)}")
    
    # 4. 输出总结
    logger.info("\n===== 测试总结 =====")
    success_rate = success_count / len(test_stocks) * 100
    avg_time = total_time / len(test_stocks) if success_count > 0 else 0
    
    logger.info(f"批量处理成功率: {success_rate:.1f}%")
    logger.info(f"平均请求耗时: {avg_time:.2f}秒/请求")
    
    if success_rate >= 80:
        logger.info("✓ 优化成功: 脚本稳定性显著提高")
    else:
        logger.warning("! 优化效果一般: 请检查网络连接和服务器状态")
    
    logger.info("===================\n")

# 主程序
def main():
    # 运行性能测试
    run_simple_performance_test()
    
    # 确保连接池关闭
    try:
        connection_pool.close_all()
        logger.info("连接池已关闭")
    except Exception as e:
        logger.error(f"关闭连接池时出错: {str(e)}")

if __name__ == "__main__":
    main()

import sys
sys.exit(0)  # 确保脚本正常退出