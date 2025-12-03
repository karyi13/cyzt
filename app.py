from flask import Flask, jsonify, render_template, request
import pandas as pd
import os
import glob
import json
import subprocess
import threading
import time
import datetime
import logging

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
KLINE_CACHE_DIR = os.path.join(DATA_DIR, 'kline_cache')

# Ensure data directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KLINE_CACHE_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data update status
update_status = {
    'is_running': False,
    'last_start_time': None,
    'last_end_time': None,
    'last_status': 'idle',
    'error': None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dates')
def get_dates():
    """Get list of available dates from limit-up files."""
    files = glob.glob(os.path.join(DATA_DIR, '*_创业板涨停.csv'))
    dates = []
    for f in files:
        basename = os.path.basename(f)
        date_str = basename.split('_')[0]
        dates.append(date_str)
    
    # Sort dates descending
    dates.sort(reverse=True)
    return jsonify(dates)

@app.route('/api/stocks/<date>')
def get_stocks(date):
    """Get stocks for a specific date with T+1 open performance."""
    file_path = os.path.join(DATA_DIR, f'{date}_创业板涨停.csv')
    if not os.path.exists(file_path):
        return jsonify({'error': 'Date not found'}), 404
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    stocks = []
    for _, row in df.iterrows():
        stock_code = str(row['股票代码'])
        stock_name = row['股票简称']
        
        # Calculate T+1 Open Performance
        kline_file = os.path.join(KLINE_CACHE_DIR, f'{date}_{stock_code}.SZ.csv')
        t1_open_pct = None
        
        if os.path.exists(kline_file):
            try:
                kline_df = pd.read_csv(kline_file)
                # Find row for Date T
                # Ensure '日期' column is string for comparison
                kline_df['日期'] = kline_df['日期'].astype(str)
                
                # Find index of the date
                matches = kline_df.index[kline_df['日期'] == date].tolist()
                
                if matches:
                    idx = matches[0]
                    # Check if T+1 exists
                    if idx + 1 < len(kline_df):
                        t_close = kline_df.iloc[idx]['收盘']
                        t1_open = kline_df.iloc[idx+1]['开盘']
                        t1_close = kline_df.iloc[idx+1]['收盘']
                        
                        if t_close != 0:
                            t1_open_pct = (t1_open - t_close) / t_close * 100
                            t1_open_pct = round(t1_open_pct, 2)
                    
                    # Check if T+2 exists for profit calculation
                    t1_open_price = None
                    t2_close_price = None
                    if idx + 1 < len(kline_df):
                        t1_open_price = kline_df.iloc[idx+1]['开盘']
                    if idx + 2 < len(kline_df):
                        t2_close_price = kline_df.iloc[idx+2]['收盘']
            except Exception as e:
                print(f"Error calculating T+1 for {stock_code}: {e}")
        
        stocks.append({
            'code': stock_code,
            'name': stock_name,
            't1_open_pct': t1_open_pct,
            't1_open_price': t1_open_price,
            't2_close_price': t2_close_price
        })
    
    return jsonify(stocks)

@app.route('/api/kline/<date>/<code>')
def get_kline(date, code):
    """Get K-line data for a specific stock."""
    kline_file = os.path.join(KLINE_CACHE_DIR, f'{date}_{code}.SZ.csv')
    if not os.path.exists(kline_file):
        return jsonify({'error': 'K-line data not found'}), 404
    
    try:
        df = pd.read_csv(kline_file)
        
        # Ensure data is sorted by date
        if '日期' in df.columns:
            df = df.sort_values('日期')
        
        # Calculate Moving Averages
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA10'] = df['收盘'].rolling(window=10).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        
        # Fill NaN with None (null in JSON) for ECharts
        df = df.where(pd.notnull(df), None)

        # Format for ECharts
        data = []
        for _, row in df.iterrows():
            data.append({
                'date': str(row['日期']),
                'open': row['开盘'],
                'close': row['收盘'],
                'low': row['最低'],
                'high': row['最高'],
                'vol': row['成交量'],
                'ma5': row['MA5'] if pd.notnull(row['MA5']) else None,
                'ma10': row['MA10'] if pd.notnull(row['MA10']) else None,
                'ma20': row['MA20'] if pd.notnull(row['MA20']) else None
            })
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_data_update_script():
    """Run the data update script in a separate process."""
    global update_status
    update_status['is_running'] = True
    update_status['last_start_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    update_status['last_status'] = 'running'
    update_status['error'] = None
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '获取K线数据_高速版.py')
    
    try:
        logger.info(f"Starting data update script: {script_path}")
        # Run the script in a separate process
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            update_status['last_status'] = 'success'
            logger.info("Data update completed successfully")
        else:
            update_status['last_status'] = 'failed'
            update_status['error'] = result.stderr
            logger.error(f"Data update failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        update_status['last_status'] = 'timeout'
        update_status['error'] = 'Script execution timed out after 1 hour'
        logger.error("Data update script timed out")
    except Exception as e:
        update_status['last_status'] = 'error'
        update_status['error'] = str(e)
        logger.error(f"Error running data update script: {str(e)}")
    finally:
        update_status['is_running'] = False
        update_status['last_end_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def background_update_thread():
    """Background thread to run data update periodically."""
    while True:
        # Check if market is closed (run after 15:30 on trading days)
        now = datetime.datetime.now()
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute
        
        # Only run on weekdays after market close (15:30)
        if weekday < 5 and hour >= 15 and minute >= 30:
            if not update_status['is_running']:
                logger.info("Scheduled data update triggered")
                run_data_update_script()
                # Sleep for 24 hours after successful update
                time.sleep(86400)
            else:
                logger.info("Update already running, skipping scheduled run")
        
        # Sleep for 1 hour before checking again
        time.sleep(3600)


@app.route('/api/update/start', methods=['POST'])
def start_data_update():
    """API endpoint to manually trigger data update."""
    if update_status['is_running']:
        return jsonify({
            'status': 'error',
            'message': 'Update already in progress'
        }), 400
    
    # Start update in a new thread
    thread = threading.Thread(target=run_data_update_script)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'success',
        'message': 'Data update started in background'
    })


@app.route('/api/update/status')
def get_update_status():
    """API endpoint to get current update status."""
    return jsonify(update_status)

@app.route('/api/last20days')
def get_last_20_days_data():
    """Get stocks from the last 20 trading days with their performance."""
    try:
        # Get all available dates sorted in descending order
        files = glob.glob(os.path.join(DATA_DIR, '*_创业板涨停.csv'))
        dates = []
        for f in files:
            basename = os.path.basename(f)
            date_str = basename.split('_')[0]
            dates.append(date_str)
        
        # Sort dates descending and take the first 20
        dates.sort(reverse=True)
        last_20_dates = dates[:20]
        
        # Collect all stocks from these dates
        all_stocks = {}
        
        for date in last_20_dates:
            file_path = os.path.join(DATA_DIR, f'{date}_创业板涨停.csv')
            if not os.path.exists(file_path):
                continue
            
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    stock_code = str(row['股票代码'])
                    stock_name = row['股票简称']
                    
                    # Calculate T+1 Open Performance
                    kline_file = os.path.join(KLINE_CACHE_DIR, f'{date}_{stock_code}.SZ.csv')
                    t1_open_pct = None
                    
                    if os.path.exists(kline_file):
                        try:
                            kline_df = pd.read_csv(kline_file)
                            kline_df['日期'] = kline_df['日期'].astype(str)
                            
                            # Find index of the date
                            matches = kline_df.index[kline_df['日期'] == date].tolist()
                            
                            if matches:
                                idx = matches[0]
                                # Check if T+1 exists
                                if idx + 1 < len(kline_df):
                                    t_close = kline_df.iloc[idx]['收盘']
                                    t1_open = kline_df.iloc[idx+1]['开盘']
                                    t1_close = kline_df.iloc[idx+1]['收盘']
                                    
                                    if t_close != 0:
                                        t1_open_pct = (t1_open - t_close) / t_close * 100
                                        t1_open_pct = round(t1_open_pct, 2)
                        except Exception as e:
                            print(f"Error calculating T+1 for {stock_code} on {date}: {e}")
                    
                    # Store stock data
                    if stock_code not in all_stocks:
                        all_stocks[stock_code] = {
                            'code': stock_code,
                            'name': stock_name,
                            'dates': []
                        }
                    
                    all_stocks[stock_code]['dates'].append({
                        'date': date,
                        't1_open_pct': t1_open_pct
                    })
            except Exception as e:
                print(f"Error reading data for {date}: {e}")
        
        # Convert to list and sort by total occurrences
        result = list(all_stocks.values())
        result.sort(key=lambda x: len(x['dates']), reverse=True)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Start background update thread
    update_thread = threading.Thread(target=background_update_thread)
    update_thread.daemon = True
    update_thread.start()
    
    logger.info("Flask app started with data update functionality")
    # Run app in production mode on cloud deployment
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

