FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app/

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建数据目录
RUN mkdir -p /app/data/kline_cache

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=5000

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["python", "app.py"]