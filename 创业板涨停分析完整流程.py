# -*- coding: utf-8 -*-
"""
创业板涨停分析完整流程
整合了三个脚本的功能：
1. 获取创业板涨停股票数据和K线数据
2. 分析涨停后表现
3. 计算仓位建议
"""

import pandas as pd
import datetime
import os
import numpy as np
import glob
from pathlib import Path
from pytdx.hq import TdxHq_API
import baostock as bs

# 延迟导入akshare，避免启动时就出现问题
try:
    import akshare as ak
except Exception as e:
    print(f"警告：导入akshare失败，可能无法获取涨停股票数据：{e}")
    ak = None

# --- 配置参数 (Configuration) ---
# 设置要获取的日期范围
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.datetime.now() - datetime.timedelta(days=20)).strftime('%Y-%m-%d')

# 数据保存目录
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
kline_dir = os.path.join(data_dir, 'kline_cache')

# 结果文件名
trade_results_filename = 'trade_results.csv'
trade_with_cangwei_filename = 'trade_with_cangwei.csv'

# --- 数据获取函数 (Data Acquisition Functions) ---
def fetch_kline_data(stock_code, date, period='daily', adjust="", max_retries=3):
    """
    获取指定股票和日期的K线数据，保存从date一年前到date后5根K线的数据
    优先使用pytdx获取，失败则使用baostock作为备选
    """
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, datetime.datetime):
        date = date.date()

    # 计算起始日期（date前一年）
    start_date_obj = date - datetime.timedelta(days=365)
    
    # 首先尝试使用pytdx获取数据
    df_pytdx = fetch_kline_with_pytdx(stock_code, date, start_date_obj)
    if df_pytdx is not None and not df_pytdx.empty:
        return df_pytdx
    
    # 如果pytdx失败，尝试使用baostock获取数据
    df_baostock = fetch_kline_with_baostock(stock_code, date, start_date_obj)
    if df_baostock is not None and not df_baostock.empty:
        return df_baostock
    
    # 如果都失败，返回空DataFrame
    return pd.DataFrame()

def fetch_kline_with_pytdx(stock_code, date, start_date_obj):
    """
    使用pytdx获取K线数据
    """
    try:
        api = TdxHq_API()
        # 连接到通达信服务器
        connected = api.connect('121.36.81.195', 7709)
        
        if not connected:
            print(f"pytdx连接失败 {stock_code}")
            return None
            
        try:
            # 将股票代码转换为通达信格式（创业板为12）
            if stock_code.startswith('30'):
                market = 12  # 创业板
            else:
                market = 1  # 深市A股
            
            # 获取K线数据（日线）
            data = api.get_security_bars(9, market, stock_code, 0, 500)  # 9=日线，500条数据
            
            if not data:
                return None
            
            # 转换为DataFrame
            df = api.to_df(data)
            
            # 转换日期格式（自动推断格式）
            df['日期'] = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')
            df['日期'] = df['日期'].dt.date
            
            # 重命名和选择列
            df = df.rename(columns={
                'open': '开盘',
                'close': '收盘',
                'high': '最高',
                'low': '最低',
                'volume': '成交量',
                'amount': '成交额'
            })
            
            # 计算振幅、涨跌幅、涨跌额、换手率（部分数据需要计算）
            df['涨跌幅'] = ((df['收盘'] - df['开盘']) / df['开盘']) * 100
            df['涨跌额'] = df['收盘'] - df['开盘']
            df['振幅'] = ((df['最高'] - df['最低']) / df['开盘']) * 100
            df['换手率'] = 0.0  # pytdx可能没有直接提供换手率
            
            # 选择需要的列
            df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']]
            
            # 将日期列转换为datetime类型以便比较
            df['日期'] = pd.to_datetime(df['日期'])
            
            # 筛选数据
            df = df[df['日期'].dt.date >= start_date_obj]
            
            if df.empty:
                return None
            
            # 找到date在K线数据中的索引
            date_index = df[df['日期'].dt.date == date].index
            
            if not date_index.empty:
                date_index = date_index[0]
                # 计算需要的结束索引（date后5根K线）
                end_data_index = date_index + 5
                # 如果结束索引超过数据长度，就用最后一根K线的索引
                end_data_index = min(end_data_index, len(df) - 1)
                # 筛选数据（从一年前到date后5根）
                df = df[:end_data_index + 1]
            
            return df
        finally:
            # 断开连接
            api.disconnect()
    except Exception as e:
        print(f"使用pytdx获取K线数据失败 {stock_code} on {date}: {e}")
        return None

def fetch_kline_with_baostock(stock_code, date, start_date_obj):
    """
    使用baostock获取K线数据
    """
    try:
        # 登录baostock
        lg = bs.login()
        if lg.error_code != '0':
            print(f"baostock登录失败: {lg.error_msg}")
            return None
        
        # 设置日期范围
        start_date = start_date_obj.strftime('%Y-%m-%d')
        end_date = (date + datetime.timedelta(days=10)).strftime('%Y-%m-%d')
        
        # 转换股票代码格式
        bs_code = f"sz.{stock_code}" if stock_code.startswith('30') else f"sz.{stock_code}"
        
        # 获取K线数据
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,preclose,volume,amount,turn",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"  # 3=不复权
        )
        
        if rs.error_code != '0':
            print(f"baostock获取数据失败: {rs.error_msg}")
            bs.logout()
            return None
        
        # 转换为DataFrame
        df = rs.get_data()
        
        # 退出登录
        bs.logout()
        
        if df.empty:
            return None
        
        # 转换日期格式
        df['日期'] = pd.to_datetime(df['date'])
        df['日期'] = df['日期'].dt.date
        
        # 转换数据类型，处理空字符串
        df['开盘'] = pd.to_numeric(df['open'], errors='coerce')
        df['收盘'] = pd.to_numeric(df['close'], errors='coerce')
        df['最高'] = pd.to_numeric(df['high'], errors='coerce')
        df['最低'] = pd.to_numeric(df['low'], errors='coerce')
        df['成交量'] = pd.to_numeric(df['volume'], errors='coerce', downcast='integer')
        df['成交额'] = pd.to_numeric(df['amount'], errors='coerce')
        df['preclose'] = pd.to_numeric(df['preclose'], errors='coerce')
        
        # 计算涨跌幅和涨跌额
        df['涨跌幅'] = ((df['收盘'] - df['preclose']) / df['preclose']) * 100
        df['涨跌额'] = df['收盘'] - df['preclose']
        
        # 去除包含NaN的行
        df = df.dropna()
        
        # 计算振幅
        df['振幅'] = ((df['最高'] - df['最低']) / df['开盘']) * 100
        
        # 重命名换手率列
        df['换手率'] = df['turn'].astype(float)
        
        # 选择需要的列
        df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']]
        
        # 将日期列转换为datetime类型以便比较
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 找到date在K线数据中的索引
        date_index = df[df['日期'].dt.date == date].index
        
        if not date_index.empty:
            date_index = date_index[0]
            # 计算需要的结束索引（date后5根K线）
            end_data_index = date_index + 5
            # 如果结束索引超过数据长度，就用最后一根K线的索引
            end_data_index = min(end_data_index, len(df) - 1)
            # 筛选数据（从一年前到date后5根）
            df = df[:end_data_index + 1]
        
        return df
    except Exception as e:
        print(f"使用baostock获取K线数据失败 {stock_code} on {date}: {e}")
        try:
            bs.logout()
        except:
            pass
        return None

def get_stock_data():
    """
    获取创业板涨停股票数据和K线数据
    """
    print(f"开始获取 {start_date} 到 {end_date} 的创业板涨停股票数据...")
    
    # 确保目录存在
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(kline_dir, exist_ok=True)
    
    # 获取当前日期
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 获取数据目录中所有创业板涨停文件
    all_cyb_files = glob.glob(os.path.join(data_dir, '*_创业板涨停.csv'))
    latest_file_date = None
    
    # 查找最新的文件日期
    if all_cyb_files:
        # 提取每个文件的日期部分并转换为datetime对象
        file_dates = []
        for file in all_cyb_files:
            try:
                # 从文件名中提取日期部分（YYYY-MM-DD格式）
                file_name = os.path.basename(file)
                date_str = file_name.split('_')[0]
                date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                file_dates.append(date_obj)
            except Exception as e:
                print(f"解析文件名日期失败 {file}: {e}")
                continue
        
        if file_dates:
            latest_file_date = max(file_dates).strftime('%Y-%m-%d')
            print(f"最新的创业板涨停文件日期是: {latest_file_date}")
    
    # 如果最新文件日期等于当前日期，则完全跳过数据更新过程
    if latest_file_date and latest_file_date == current_date:
        print(f"最新文件日期 {latest_file_date} 与当前日期 {current_date} 相同，完全跳过数据更新过程")
        return True
    
    # 获取交易日历
    if ak is None:
        print("错误：akshare未导入成功，无法获取交易日历数据")
        return False
    
    trade_dates_df = ak.tool_trade_date_hist_sina()
    trade_dates_df['trade_date'] = pd.to_datetime(trade_dates_df['trade_date'])
    
    # 筛选出指定日期范围内的交易日
    filtered_trade_dates = trade_dates_df[
        (trade_dates_df['trade_date'] >= start_date) &
        (trade_dates_df['trade_date'] <= end_date)
    ]
    
    all_filtered_stocks = []
    
    # 遍历每个交易日
    for index, row in filtered_trade_dates.iterrows():
        current_date_str = row['trade_date'].strftime('%Y-%m-%d')
        print(f"处理日期: {current_date_str}")
        
        # 不再检查文件是否存在，直接更新数据（覆盖旧文件）
        # 确保即使存在旧文件，也会更新最近20天的数据
        # if os.path.exists(file_path):
        #    print(f"{current_date_str}_创业板涨停.csv 文件已存在，跳过该日期的数据更新")
        #    continue
        
        # 不再单独检查当前日期与最新文件日期的比较
        # if latest_file_date and current_date_str == latest_file_date:
        #    print(f"当前日期 {current_date_str} 与最新文件日期 {latest_file_date} 相同，跳过数据更新")
        #    continue
    
        try:
            # 获取当日涨停股票
            if ak is None:
                print("错误：akshare未导入成功，无法获取涨停股票数据")
                continue
                
            daily_limit_up_stocks_df = ak.stock_zt_pool_em(current_date_str.replace("-", ""))
            print(f"获取到 {len(daily_limit_up_stocks_df)} 只涨停股票")
    
            # 筛选创业板（代码以30开头）且连板数为1的股票
            zt_cyb_filtered_daily = daily_limit_up_stocks_df[
                (daily_limit_up_stocks_df['代码'].str.startswith('30')) &
                (daily_limit_up_stocks_df['连板数'] == 1)
            ]
    
            print(f"筛选出 {len(zt_cyb_filtered_daily)} 只创业板首板股票")
    
            if not zt_cyb_filtered_daily.empty:
                # 选择需要的列
                zt_cyb_selected_daily = zt_cyb_filtered_daily[['代码', '名称']].copy()
                
                # 添加日期列
                zt_cyb_selected_daily['日期'] = current_date_str
                
                # 重命名列
                zt_cyb_selected_daily.rename(columns={'代码': '股票代码', '名称': '股票简称'}, inplace=True)
                
                # 添加到列表
                all_filtered_stocks.append(zt_cyb_selected_daily)
                
                # 保存到CSV文件
                file_path = os.path.join(data_dir, f'{current_date_str}_创业板涨停.csv')
                zt_cyb_selected_daily.to_csv(file_path, index=False, encoding='utf-8')
                print(f"数据已保存到: {file_path}")
                
                # 获取并保存K线数据
                for _, stock_row in zt_cyb_selected_daily.iterrows():
                    stock_code = stock_row['股票代码']
                    stock_name = stock_row['股票简称']
                    print(f"  获取 {stock_name}({stock_code}) 的K线数据")
                    
                    # 检查文件是否已存在
                    kline_file_path = os.path.join(kline_dir, f'{current_date_str}_{stock_code}.SZ.csv')
                    need_to_fetch = True
                    
                    if os.path.exists(kline_file_path):
                        # 读取已保存的文件
                        existing_kline_df = pd.read_csv(kline_file_path, encoding='utf-8')
                        if not existing_kline_df.empty:
                            # 将日期列转换为datetime类型以便比较
                            existing_kline_df['日期'] = pd.to_datetime(existing_kline_df['日期'])
                            
                            # 检查最新数据是否是最新交易日数据
                            latest_data_date = existing_kline_df['日期'].max().date()
                            current_date = datetime.datetime.now().date()
                            
                            # 如果最新数据距离当前日期超过2天，需要重新获取
                            if (current_date - latest_data_date).days <= 2:
                                print(f"  文件已存在且数据较新，跳过获取: {kline_file_path}")
                                need_to_fetch = False
                            else:
                                print(f"  文件已存在但数据过时，重新获取: {kline_file_path}")
                    
                    if need_to_fetch:
                        # 获取K线数据
                        kline_df = fetch_kline_data(stock_code, current_date_str)
                        
                        if not kline_df.empty:
                            # 保存K线数据
                            kline_df.to_csv(kline_file_path, index=False, encoding='utf-8')
                            print(f"  K线数据已保存到: {kline_file_path}")
                        else:
                            print(f"  未获取到 {stock_name}({stock_code}) 的K线数据")
                            # 创建空的K线文件，确保所有股票都有记录
                            with open(kline_file_path, 'w', encoding='utf-8') as f:
                                f.write('日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率\n')
                            print(f"  已创建空的K线文件: {kline_file_path}")
            else:
                print(f"当日没有符合条件的创业板首板股票")
                # 创建空文件
                file_path = os.path.join(data_dir, f'{current_date_str}_创业板涨停.csv')
                pd.DataFrame(columns=['股票代码', '股票简称', '日期']).to_csv(file_path, index=False, encoding='utf-8')
                print(f"已创建空文件: {file_path}")
        
        except Exception as e:
            print(f"处理日期 {current_date_str} 时发生错误: {e}")
            continue
    
    # 合并所有数据
    if all_filtered_stocks:
        final_filtered_df = pd.concat(all_filtered_stocks, ignore_index=True)
        print(f"\n总共有 {len(final_filtered_df)} 只股票符合条件")
    else:
        print("\n没有符合条件的股票数据")
    
    print("\n数据获取完成！")
    return True

# --- 涨停表现分析函数 (Performance Analysis Functions) ---
def analyze_trade_performance():
    """
    分析涨停后的交易表现，生成交易结果
    """
    print("\n--- 开始分析涨停后表现 ---")
    
    # 检查KLINE_DIR是否存在
    if not os.path.isdir(kline_dir):
        print(f"错误：KLINE_DIR '{kline_dir}' 不存在。请检查路径或创建该目录。")
        return False
    
    # 读取所有CSV文件到pandas DataFrames
    kline_files = list(Path(kline_dir).glob("*.csv"))
    
    if not kline_files:
        print(f"在目录 '{kline_dir}' 中没有找到任何CSV文件。请确保您的CSV文件在此目录中。")
        return False
    
    all_stock_data = {}
    for file_path in kline_files:
        file_name = file_path.stem  # 获取文件名（不含扩展名）
        try:
            df = pd.read_csv(file_path)
            all_stock_data[file_name] = df
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}。跳过此文件。")
    
    if not all_stock_data:
        print("没有成功读取任何股票数据。请检查您的CSV文件格式。")
        return False
    
    # 处理数据并计算交易结果
    trade_results = []
    
    for file_name, df in all_stock_data.items():
        parts = file_name.split("_")
        if len(parts) < 2:  # 检查文件名格式是否符合 '日期_股票代码.csv'
            print(f"文件名格式不正确，跳过文件: {file_name}。期望格式为 'YYYY-MM-DD_股票代码.csv'。")
            continue
        
        date_str = parts[0]
        stock_code = parts[1].split(".")[0]  # 去除可能的后缀，如.SZ
        
        if '日期' not in df.columns:  # 检查 '日期' 列是否存在
            print(f"股票代码: {stock_code}, 涨停日期: {date_str}: 数据中没有找到 '日期' 列，跳过处理。")
            continue
        
        df['日期'] = pd.to_datetime(df['日期'])  # 将 '日期' 列转换为 datetime 对象
        try:
            trade_date = pd.to_datetime(date_str)
        except ValueError:
            print(f"文件名中的日期格式不正确，跳过文件: {file_name}。期望格式为 'YYYY-MM-DD'。")
            continue
        
        # 查找涨停日期在DataFrame中的索引
        if trade_date in df['日期'].values:
            trade_date_index = df[df['日期'] == trade_date].index[0]
            buy_date_index = trade_date_index + 1  # 买入日期是涨停日期的下一个交易日
            sell_date_index = buy_date_index + 1   # 卖出日期是买入日期的下一个交易日
        
            # 检查是否有足够的后续交易日数据
            if sell_date_index < len(df):
                buy_price = df.loc[buy_date_index, '开盘']  # 买入价是下一个交易日开盘价
                sell_price = df.loc[sell_date_index, '收盘']  # 卖出价是买入日期的下一个交易日收盘价
                profit_loss = sell_price - buy_price
        
                # 计算买入日期和卖出日期的涨跌幅
                buy_date_open_change = None
                buy_date_close_change = None
                sell_date_open_change = None
                sell_date_close_change = None
        
                # 涨跌幅计算需要前一日的收盘价
                if trade_date_index >= 0:
                    prev_close_price_buy = df.loc[trade_date_index, '收盘']
                    if prev_close_price_buy != 0:  # 避免除以零
                        buy_date_open_change = ((df.loc[buy_date_index, '开盘'] - prev_close_price_buy) / prev_close_price_buy) * 100
                        buy_date_close_change = ((df.loc[buy_date_index, '收盘'] - prev_close_price_buy) / prev_close_price_buy) * 100
                    else:
                        print(f"股票代码: {stock_code}, 涨停日期: {trade_date.date()}: 前一日收盘价为零，无法计算买入日涨跌幅。")
        
                if buy_date_index >= 0 and buy_date_index < len(df):
                    prev_close_price_sell = df.loc[buy_date_index, '收盘']
                    if prev_close_price_sell != 0:  # 避免除以零
                        sell_date_open_change = ((df.loc[sell_date_index, '开盘'] - prev_close_price_sell) / prev_close_price_sell) * 100
                        sell_date_close_change = ((df.loc[sell_date_index, '收盘'] - prev_close_price_sell) / prev_close_price_sell) * 100
                    else:
                        print(f"股票代码: {stock_code}, 涨停日期: {trade_date.date()}: 买入日收盘价为零，无法计算卖出日涨跌幅。")
        
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
            else:
                print(f"股票代码: {stock_code}, 涨停日期: {trade_date.date()}: 没有足够的后续交易日来确定卖出价格和涨跌幅数据。")
        else:
            print(f"股票代码: {stock_code}, 涨停日期: {trade_date.date()}: 涨停日期不在数据范围内。")
    
    # 创建DataFrame并计算 '盈亏比例'
    if trade_results:
        trade_df = pd.DataFrame(trade_results)
        
        # 计算 '盈亏比例'
        # 避免买入价为0导致除以零错误
        trade_df['盈亏比例'] = trade_df.apply(lambda row: ((row['卖出价'] - row['买入价']) / row['买入价']) * 100 if row['买入价'] != 0 else 0, axis=1)
        
        # 保存到CSV文件
        output_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), trade_results_filename)
        try:
            trade_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')  # 使用utf-8-sig编码避免中文乱码
            print(f"\n交易结果已保存到：{output_csv_path}")
            print("\n计算结果前5行:")
            print(trade_df.head().to_markdown(index=False))
            print("\nDataFrame信息:")
            trade_df.info()
            return True
        except Exception as e:
            print(f"保存CSV文件失败: {e}")
            return False
    else:
        print("没有生成任何交易结果数据。请检查您的输入文件和处理逻辑。")
        return False

# --- 仓位建议计算函数 (Position Sizing Functions) ---
def calculate_position_suggestions():
    """
    计算仓位建议
    """
    print("\n--- 开始计算仓位建议 ---")
    
    # 读取交易结果文件
    input_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), trade_results_filename)
    
    if not os.path.exists(input_file_path):
        print(f"错误：输入文件 '{input_file_path}' 不存在")
        return False
    
    df = pd.read_csv(input_file_path)
    print(f"CSV文件 '{input_file_path}' 读取成功。")
    
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
    
    # Handle cases where there are no winning or losing trades to avoid division by zero or infinity
    grouped_df['平均盈利'] = grouped_df['平均盈利'].replace([np.inf, -np.inf], np.nan).fillna(0)
    grouped_df['平均亏损'] = grouped_df['平均亏损'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate the Win/Loss Ratio, handle division by zero if no losing trades
    grouped_df['盈亏比'] = np.where(grouped_df['平均亏损'] != 0, abs(grouped_df['平均盈利'] / grouped_df['平均亏损']), np.nan)
    
    # Calculate Kelly Percentage, handle division by zero or NaN values
    grouped_df['凯莉公式仓位'] = np.where(
        (grouped_df['盈亏比'].notna()) & (grouped_df['胜率'].notna()),
        grouped_df['胜率'] - (1 - grouped_df['胜率']) / grouped_df['盈亏比'],
        np.nan
    )
    
    # Filter out rows where '总交易数' is 0
    filtered_grouped_df = grouped_df[grouped_df['总交易数'] != 0].copy()
    
    # Merge the original df with the filtered_grouped_df, including the '胜率' column
    merged_df = pd.merge(df, filtered_grouped_df[['排名', '涨跌幅区间', '凯莉公式仓位', '胜率']],
                         on=['排名', '涨跌幅区间'],
                         how='left')
    
    # Define a function to calculate the position size based on the specified trading rules
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
    
    # Save the merged_df to CSV
    output_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), trade_with_cangwei_filename)
    merged_df.to_csv(output_csv_path, index=False)
    print(f"\n仓位建议已保存到: {output_csv_path}")
    
    return True

# --- 主函数 (Main Function) ---
def main():
    """
    主函数，按顺序执行所有步骤
    """
    print("=== 创业板涨停分析完整流程开始 ===")
    
    # 步骤1: 获取数据
    if not get_stock_data():
        print("数据获取失败，流程终止")
        return
    
    # 步骤2: 分析涨停表现
    if not analyze_trade_performance():
        print("涨停表现分析失败，流程终止")
        return
    
    # 步骤3: 计算仓位建议
    if not calculate_position_suggestions():
        print("仓位建议计算失败，流程终止")
        return
    
    print("\n=== 创业板涨停分析完整流程完成 ===")
    print(f"最终结果文件: {os.path.join(os.path.dirname(os.path.abspath(__file__)), trade_with_cangwei_filename)}")

if __name__ == "__main__":
    main()
