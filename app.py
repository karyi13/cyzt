from flask import Flask, jsonify, render_template, request, send_from_directory
import os
import glob
import csv
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

# In a serverless environment, we may not have persistent storage
# So check if directories exist before creating them
if not os.environ.get('VERCEL'):
    # Only create directories during local development
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(KLINE_CACHE_DIR, exist_ok=True)
else:
    # In Vercel environment, we'll check if directories exist at runtime
    pass  # Vercel will have directories created if files are deployed

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

@app.route('/index_new.html')
def index_new():
    return render_template('index_new.html')

@app.route('/mobile')
def mobile():
    return render_template('mobile.html')

@app.route('/mobile_optimized')
def mobile_optimized():
    return render_template('mobile_optimized.html')

@app.route('/stock_analysis_enhanced_v2.html')
def stock_analysis_enhanced():
    return render_template('stock_analysis_enhanced_v2.html')

@app.route('/trade_with_cangwei.csv')
def get_trade_csv():
    """Serve the trade_with_cangwei.csv file."""
    return send_from_directory('templates', 'trade_with_cangwei.csv', mimetype='text/csv')

@app.route('/data/kline_cache/<path:filename>')
def serve_kline_cache(filename):
    """Serve K-line cache CSV files."""
    return send_from_directory(KLINE_CACHE_DIR, filename, mimetype='text/csv')

@app.route('/api/dates')
def get_dates():
    """Get list of available dates from limit-up files."""
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        return jsonify([])  # Return empty list if no data directory

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
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        return jsonify({'error': 'Data directory not found'}), 404

    file_path = os.path.join(DATA_DIR, f'{date}_创业板涨停.csv')
    if not os.path.exists(file_path):
        return jsonify({'error': 'Date not found'}), 404

    stocks = []

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # 使用utf-8-sig处理BOM
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            # 处理可能包含BOM的列名
            stock_code_key = '股票代码'
            stock_name_key = '股票简称'

            # 检查列名是否包含BOM或其他特殊字符
            if '股票代码' not in reader.fieldnames:
                # 尝试找到包含'股票代码'的列名
                for field in reader.fieldnames:
                    if '股票代码' in field or field.replace('\ufeff', '').strip() == '股票代码':
                        stock_code_key = field
                        break

            if '股票简称' not in reader.fieldnames:
                # 尝试找到包含'股票简称'的列名
                for field in reader.fieldnames:
                    if '股票简称' in field or field.replace('\ufeff', '').strip() == '股票简称':
                        stock_name_key = field
                        break

            stock_code = str(row[stock_code_key])
            stock_name = row[stock_name_key]

            # Calculate T+1 Open Performance
            # Remove existing .SZ or .SS suffix from stock_code to avoid duplicate suffixes
            stock_code_clean = stock_code.replace('.SZ', '').replace('.SS', '')
            kline_file = os.path.join(KLINE_CACHE_DIR, f'{date}_{stock_code_clean}.SZ.csv')
            t1_open_pct = None
            t1_open_price = None
            t2_close_price = None

            if os.path.exists(kline_file):
                try:
                    # Read the K-line file and find the date
                    with open(kline_file, 'r', encoding='utf-8') as kf:
                        kline_reader = csv.DictReader(kf)
                        kline_rows = list(kline_reader)

                    # Find index of the date
                    matched_idx = -1
                    for idx, k_row in enumerate(kline_rows):
                        if k_row['日期'] == date:
                            matched_idx = idx
                            break

                    if matched_idx != -1:
                        # Check if T+1 exists
                        if matched_idx + 1 < len(kline_rows):
                            # Handle potential empty or invalid values
                            t_close_str = kline_rows[matched_idx]['收盘']
                            t1_open_str = kline_rows[matched_idx+1]['开盘']

                            if t_close_str and t_close_str != '' and t1_open_str and t1_open_str != '':
                                try:
                                    t_close = float(t_close_str)
                                    t1_open = float(t1_open_str)

                                    if t_close != 0:
                                        t1_open_pct = (t1_open - t_close) / t_close * 100
                                        t1_open_pct = round(t1_open_pct, 2)

                                    t1_open_price = t1_open
                                except (ValueError, TypeError):
                                    print(f"Invalid data for T+1 calculation for {stock_code}: close={t_close_str}, open={t1_open_str}")

                        # Check if T+2 exists for profit calculation
                        if matched_idx + 2 < len(kline_rows):
                            t2_close_str = kline_rows[matched_idx+2]['收盘']
                            if t2_close_str and t2_close_str != '':
                                try:
                                    t2_close_price = float(t2_close_str)
                                except (ValueError, TypeError):
                                    print(f"Invalid data for T+2 calculation for {stock_code}: close={t2_close_str}")

                except Exception as e:
                    print(f"Error calculating T+1 for {stock_code}: {e}")

            stocks.append({
                'code': stock_code,
                'name': stock_name,
                't1_open_pct': t1_open_pct,
                't1_open_price': t1_open_price,
                't2_close_price': t2_close_price
            })

        # 按照T+1开盘涨跌幅降序排序
        stocks.sort(key=lambda x: x['t1_open_pct'] if x['t1_open_pct'] is not None else -float('inf'), reverse=True)

        return jsonify(stocks)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/kline/<date>/<code>')
def get_kline(date, code):
    """Get K-line data for a specific stock."""
    # Check if kline cache directory exists
    if not os.path.exists(KLINE_CACHE_DIR):
        return jsonify({'error': 'K-line cache directory not found'}), 404

    # Remove existing .SZ or .SS suffix from code to avoid duplicate suffixes
    code_clean = code.replace('.SZ', '').replace('.SS', '')
    kline_file = os.path.join(KLINE_CACHE_DIR, f'{date}_{code_clean}.SZ.csv')
    if not os.path.exists(kline_file):
        return jsonify({'error': 'K-line data not found'}), 404

    try:
        with open(kline_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Sort data by date if '日期' column exists
        if '日期' in reader.fieldnames:
            rows.sort(key=lambda x: x['日期'])  # Sorting by date string

        # Calculate Moving Averages - this requires a bit more work without pandas
        for i, row in enumerate(rows):
            # Convert values to float for calculation, handling empty/None values
            try:
                row['收盘'] = float(row['收盘']) if row['收盘'] and row['收盘'] != '' else 0.0
            except (ValueError, TypeError):
                row['收盘'] = 0.0

        # Calculate moving averages - simplified implementation
        for i, row in enumerate(rows):
            # Calculate MA5 for current position
            if i >= 4:
                valid_values = []
                for j in range(i-4, i+1):
                    try:
                        val = float(rows[j]['收盘']) if rows[j]['收盘'] and rows[j]['收盘'] != '' else 0.0
                        if val > 0:  # Only include valid positive values
                            valid_values.append(val)
                    except (ValueError, TypeError):
                        continue
                if len(valid_values) == 5:  # Only calculate if all 5 values are valid
                    row['MA5'] = sum(valid_values) / 5
                else:
                    row['MA5'] = None
            else:
                row['MA5'] = None

            # Calculate MA10 for current position
            if i >= 9:
                valid_values = []
                for j in range(i-9, i+1):
                    try:
                        val = float(rows[j]['收盘']) if rows[j]['收盘'] and rows[j]['收盘'] != '' else 0.0
                        if val > 0:  # Only include valid positive values
                            valid_values.append(val)
                    except (ValueError, TypeError):
                        continue
                if len(valid_values) == 10:  # Only calculate if all 10 values are valid
                    row['MA10'] = sum(valid_values) / 10
                else:
                    row['MA10'] = None
            else:
                row['MA10'] = None

            # Calculate MA20 for current position
            if i >= 19:
                valid_values = []
                for j in range(i-19, i+1):
                    try:
                        val = float(rows[j]['收盘']) if rows[j]['收盘'] and rows[j]['收盘'] != '' else 0.0
                        if val > 0:  # Only include valid positive values
                            valid_values.append(val)
                    except (ValueError, TypeError):
                        continue
                if len(valid_values) == 20:  # Only calculate if all 20 values are valid
                    row['MA20'] = sum(valid_values) / 20
                else:
                    row['MA20'] = None
            else:
                row['MA20'] = None

        # Format for ECharts
        data = []
        for row in rows:
            try:
                data.append({
                    'date': str(row['日期']),
                    'open': float(row['开盘']) if row['开盘'] and row['开盘'] != '' else None,
                    'close': float(row['收盘']) if row['收盘'] and row['收盘'] != '' else None,
                    'low': float(row['最低']) if row['最低'] and row['最低'] != '' else None,
                    'high': float(row['最高']) if row['最高'] and row['最高'] != '' else None,
                    'vol': int(float(row['成交量'])) if row['成交量'] and row['成交量'] != '' else None,
                    'ma5': row['MA5'],
                    'ma10': row['MA10'],
                    'ma20': row['MA20']
                })
            except (ValueError, TypeError):
                # Skip rows with invalid data
                continue
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

    # Only run this in development, not on Vercel
    # On Vercel, data should be pre-populated or updated through a different mechanism
    if os.environ.get('VERCEL'):
        update_status['last_status'] = 'skipped'
        update_status['error'] = 'Data update skipped on Vercel deployment'
        logger.info("Skipping data update on Vercel")
        update_status['is_running'] = False
        update_status['last_end_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return

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
    # Don't run background updates on Vercel
    if os.environ.get('VERCEL'):
        logger.info("Background updates disabled on Vercel")
        return

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
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        return jsonify({'error': 'Data directory not found'}), 404

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
                with open(file_path, 'r', encoding='utf-8-sig') as f:  # 使用utf-8-sig处理BOM
                    reader = csv.DictReader(f)
                    rows = list(reader)

                for row in rows:
                    # 处理可能包含BOM的列名
                    stock_code_key = '股票代码'
                    stock_name_key = '股票简称'

                    # 检查列名是否包含BOM或其他特殊字符
                    if '股票代码' not in reader.fieldnames:
                        # 尝试找到包含'股票代码'的列名
                        for field in reader.fieldnames:
                            if '股票代码' in field or field.replace('\ufeff', '').strip() == '股票代码':
                                stock_code_key = field
                                break

                    if '股票简称' not in reader.fieldnames:
                        # 尝试找到包含'股票简称'的列名
                        for field in reader.fieldnames:
                            if '股票简称' in field or field.replace('\ufeff', '').strip() == '股票简称':
                                stock_name_key = field
                                break

                    stock_code = str(row[stock_code_key])
                    stock_name = row[stock_name_key]

                    # Calculate T+1 Open Performance
                    # Remove existing .SZ or .SS suffix from stock_code to avoid duplicate suffixes
                    stock_code_clean = stock_code.replace('.SZ', '').replace('.SS', '')
                    kline_file = os.path.join(KLINE_CACHE_DIR, f'{date}_{stock_code_clean}.SZ.csv')
                    t1_open_pct = None

                    if os.path.exists(kline_file):
                        try:
                            # Read the K-line file
                            with open(kline_file, 'r', encoding='utf-8') as kf:
                                kline_reader = csv.DictReader(kf)
                                kline_rows = list(kline_reader)

                            # Find index of the date
                            matched_idx = -1
                            for idx, k_row in enumerate(kline_rows):
                                if k_row['日期'] == date:
                                    matched_idx = idx
                                    break

                            if matched_idx != -1:
                                # Check if T+1 exists
                                if matched_idx + 1 < len(kline_rows):
                                    t_close_str = kline_rows[matched_idx]['收盘']
                                    t1_open_str = kline_rows[matched_idx+1]['开盘']

                                    if t_close_str and t_close_str != '' and t1_open_str and t1_open_str != '':
                                        try:
                                            t_close = float(t_close_str)
                                            t1_open = float(t1_open_str)

                                            if t_close != 0:
                                                t1_open_pct = (t1_open - t_close) / t_close * 100
                                                t1_open_pct = round(t1_open_pct, 2)
                                        except (ValueError, TypeError):
                                            print(f"Invalid data for T+1 calculation for {stock_code} on {date}: close={t_close_str}, open={t1_open_str}")
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
    # In a serverless environment like Vercel, we don't run background threads
    # Only start background update thread in local development
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--local':
        update_thread = threading.Thread(target=background_update_thread)
        update_thread.daemon = True
        update_thread.start()
        logger.info("Background update thread started for local development")
    elif os.environ.get('VERCEL'):
        # Don't start background thread on Vercel
        logger.info("Running on Vercel, background updates disabled")

    logger.info("Flask app started")
    # Run app in production mode on cloud deployment
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

# For Vercel deployment, make sure to import app at module level
# This allows Vercel's Python runtime to correctly load the application
elif __name__.startswith('vc_apprunner_') or os.environ.get('VERCEL'):
    # When running in Vercel environment
    logger.info("Vercel environment detected, initializing app")

