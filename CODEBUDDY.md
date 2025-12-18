# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Project Overview

创业板涨停复盘Web应用 - A Flask-based web application for analyzing limit-up stocks on China's ChiNext board (创业板). The application fetches daily limit-up stock data, downloads K-line (candlestick) data, and provides visualizations and T+1 opening performance analysis.

**Tech Stack**: Python Flask, Pandas, akshare, pytdx, baostock, ECharts

## Common Development Commands

### Running the Application

```bash
# Run the Flask web application (default port 5000)
python app.py
```

The application will:
- Start a background thread for automatic data updates (weekdays after 15:30)
- Serve the web interface on `http://0.0.0.0:5000`
- Listen on port specified by `PORT` environment variable if set

### Data Update Script

```bash
# Manually run the data update script
python 获取K线数据_高速版.py
```

This script:
- Fetches limit-up stocks from akshare for the configured date range
- Downloads K-line data using pytdx (primary) or baostock (fallback)
- Saves data to `./data/` directory with CSV files named `YYYY-MM-DD_创业板涨停.csv`
- Caches K-line data in `./data/kline_cache/` with format `YYYY-MM-DD_STOCKCODE.SZ.csv`

### Alternative Analysis Script

```bash
# Run the complete analysis workflow
python 创业板涨停分析完整流程.py
```

This integrates data fetching, analysis, and position calculation in one script.

### Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

## High-Level Architecture

### Application Structure

```
app.py (Flask Application)
├── Routes
│   ├── / → index.html (主页面)
│   ├── /mobile → mobile.html (移动端页面)
│   ├── /api/dates → Get available trading dates
│   ├── /api/stocks/<date> → Get stocks with T+1 performance for a date
│   ├── /api/kline/<date>/<code> → Get K-line data with MA indicators
│   ├── /api/last20days → Get stocks from last 20 trading days
│   ├── /api/update/start → Trigger manual data update (POST)
│   └── /api/update/status → Check update status
└── Background Thread
    └── Auto-update scheduler (runs on weekdays after 15:30)
```

### Data Flow

1. **Data Collection** (`获取K线数据_高速版.py`):
   - Fetch limit-up stock list from akshare for date range
   - For each stock, download K-line data (1 year prior to limit-up date + 30 days after)
   - Save to CSV files in `./data/` and `./data/kline_cache/`

2. **Data Serving** (`app.py`):
   - Read CSV files from `./data/` directory
   - Calculate T+1 opening performance by comparing closing price on limit-up day with next day's opening
   - Compute Moving Averages (MA5, MA10, MA20) for K-line charts
   - Return JSON data to frontend

3. **Frontend Visualization** (`templates/index.html`, `templates/mobile.html`):
   - Display stock list with T+1 performance sorted by opening percentage
   - Render K-line charts using ECharts library
   - Show performance statistics and analysis

### Data Update Strategy

The application uses a **dual-source failover** approach for K-line data:

1. **Primary: pytdx** (通达信协议)
   - Fast, local protocol
   - Connection pool management (`PytdxConnectionPool` class)
   - Configured with fallback servers
   - Market code: 0 for stocks starting with `00` or `30` (深市/创业板), 1 otherwise

2. **Fallback: baostock**
   - Used when pytdx fails
   - API-based data source
   - Requires login/logout session management

Both sources fetch:
- **Time Range**: From 1 year before the limit-up date to 30 days after
- **Data Retention**: Only keeps data up to 5 bars after the target date
- **Fields**: 日期, 开盘, 最高, 最低, 收盘, 成交量, 成交额, 股票代码

### Key Data Calculations

**T+1 Opening Performance** (app.py:74-98):
```python
# Find the limit-up date (T) in K-line data
# Calculate: (T+1 opening price - T closing price) / T closing price * 100
t1_open_pct = (t1_open - t_close) / t_close * 100
```

**Moving Averages** (app.py:137-140):
```python
df['MA5'] = df['收盘'].rolling(window=5).mean()
df['MA10'] = df['收盘'].rolling(window=10).mean()
df['MA20'] = df['收盘'].rolling(window=20).mean()
```

### Important Configuration

**Date Range** (in `获取K线数据_高速版.py`):
- `start_date = '2025-11-20'` (configurable)
- `end_date = datetime.datetime.now().strftime('%Y-%m-%d')` (auto-updated to today)

**Data Directories**:
- `./data/` - Daily limit-up CSV files
- `./data/kline_cache/` - K-line data cache

**Connection Pool Settings** (获取K线数据_高速版.py:86-106):
- Max connections: 10
- Health check interval: 180 seconds
- Timeout: 30 seconds per request
- Retry strategy: Built-in with exponential backoff

### Network Configuration

The script includes special proxy/environment handling:
- Disables all proxy environment variables
- Configures requests session without proxy
- Custom User-Agent headers for akshare compatibility

### Background Update Thread

**Automatic Updates** (app.py:205-225):
- Runs continuously in daemon thread
- Triggers on weekdays (Monday-Friday) after 15:30
- Executes `获取K线数据_高速版.py` as subprocess
- 1-hour timeout for script execution
- 24-hour sleep after successful update

**Update Status Tracking** (app.py:26-33):
```python
update_status = {
    'is_running': bool,
    'last_start_time': timestamp,
    'last_end_time': timestamp,
    'last_status': 'idle'|'running'|'success'|'failed'|'timeout'|'error',
    'error': error_message
}
```

## File Organization Patterns

- Stock data files: `YYYY-MM-DD_创业板涨停.csv`
- K-line cache files: `YYYY-MM-DD_STOCKCODE.SZ.csv`
- All ChiNext stocks use `.SZ` suffix (Shenzhen exchange)
- Stock codes starting with `30` are ChiNext stocks

## Testing and Development

- The application has no formal test suite
- Test by running the Flask app and checking API endpoints manually
- Monitor `kline_fetch.log` and `error_log.txt` for data fetching issues
- Check browser console for frontend errors

## Important Notes

- **Market Data**: Only fetches stocks with `连板数 == 1` (first-time limit-up only)
- **Timezone**: All times are local system time (likely China Standard Time)
- **Data Dependencies**: Requires working akshare, pytdx, and baostock connections
- **Windows Compatibility**: Paths use `os.path.join()` for cross-platform support
- **Threading**: Background update thread is daemon - stops when main app stops
