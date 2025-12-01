#!/bin/bash

# 创业板涨停复盘Web应用部署脚本
echo "开始部署创业板涨停复盘应用..."

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p data/kline_cache

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 设置环境变量
export FLASK_APP=app.py
export FLASK_ENV=production

# 如果有.env文件，加载环境变量
if [ -f .env ]; then
    echo "加载环境变量..."
    export $(grep -v '^#' .env | xargs)
fi

# 启动应用
echo "启动Flask应用..."
python app.py