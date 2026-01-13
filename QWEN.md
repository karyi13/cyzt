# 创业板涨停分析系统

## 项目概述

这是一个基于Python Flask的Web应用程序，专门用于分析A股创业板的涨停股票数据。系统能够获取创业板涨停股票数据、K线数据，并提供历史数据分析、T+1表现分析、仓位建议等功能。

## 项目架构

### 核心组件

1. **数据获取模块** (`获取K线数据_高速版.py`)
   - 使用akshare获取涨停池数据
   - 通过pytdx和baostock获取K线数据
   - 实现了连接池和重试机制以提高稳定性

2. **数据分析模块** (`创业板涨停分析完整流程.py`)
   - 获取创业板涨停股票数据和K线数据
   - 分析涨停后表现
   - 计算仓位建议（基于凯莉公式）

3. **Web应用模块** (`app.py`)
   - Flask后端API服务
   - 提供数据接口（日期列表、股票数据、K线数据等）
   - 支持Vercel部署

4. **前端界面** (`templates/index.html`)
   - 响应式Web界面
   - ECharts图表展示K线数据
   - 支持主题切换（亮色/暗色模式）
   - 提供对比模式和20日数据查看

### 数据结构

- **数据目录**: `data/`
  - 涨停股票数据: `YYYY-MM-DD_创业板涨停.csv`
  - K线缓存数据: `kline_cache/` 目录下
- **模板目录**: `templates/`
  - 前端HTML文件
  - 分析结果CSV文件

## 技术栈

- **后端**: Python Flask
- **数据获取**: akshare, pytdx, baostock
- **数据处理**: pandas, numpy
- **前端**: HTML/CSS/JavaScript, ECharts
- **部署**: Vercel (支持云部署)

## 功能特性

1. **数据获取与更新**
   - 自动获取创业板涨停股票数据
   - 获取对应K线数据
   - 支持手动和自动数据更新

2. **数据分析**
   - 涨停后T+1表现分析
   - 仓位建议计算
   - 历史数据回测

3. **Web界面**
   - 日期选择器
   - 股票列表展示
   - K线图表可视化
   - 主题切换功能
   - 移动端适配

## 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖:
- Flask==2.3.3
- akshare>=1.10.50
- pandas>=1.5.0
- pytdx>=1.70
- baostock>=0.8.8
- numpy>=1.21.0

## 运行方式

### 本地开发
```bash
python app.py --local
```

### 生产部署
项目已配置为可在Vercel上部署，使用`wsgi.py`作为入口点。

## API接口

- `GET /api/dates` - 获取可用日期列表
- `GET /api/stocks/<date>` - 获取指定日期的涨停股票
- `GET /api/kline/<date>/<code>` - 获取指定股票的K线数据
- `GET /api/last20days` - 获取最近20日数据
- `POST /api/update/start` - 手动启动数据更新
- `GET /api/update/status` - 获取更新状态

## 部署配置

- **Vercel部署**: 通过`vercel.json`配置
- **WSGI入口**: `wsgi.py`
- **环境变量**: 支持PORT设置

## 项目特点

1. **数据完整性**: 包含从2025年1月到12月的完整创业板涨停数据
2. **高性能**: 使用连接池和缓存机制优化数据获取
3. **可扩展性**: 模块化设计，易于扩展新功能
4. **云原生**: 支持Vercel等云平台部署
5. **用户友好**: 提供直观的Web界面和多种分析视图