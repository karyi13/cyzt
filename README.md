# 创业板涨停复盘Web应用

## 项目简介

这是一个基于Flask框架开发的创业板涨停复盘Web应用，提供股票数据可视化和分析功能，支持K线图表展示和次日开盘表现统计。

## 功能特性

- 📊 创业板涨停股票数据展示
- 📈 K线图表可视化（支持MA均线）
- 📉 T+1开盘表现分析
- 🔄 自动数据更新功能（交易日收盘后自动更新）
- ⏰ 手动触发数据更新API

## 技术栈

- **后端**: Python Flask
- **数据处理**: Pandas
- **前端**: HTML, CSS, JavaScript, ECharts
- **数据来源**: akshare, pytdx, baostock

## 项目结构

```
├── app.py                 # Flask应用主文件
├── 获取K线数据_高速版.py     # 数据更新脚本
├── requirements.txt       # 项目依赖
├── start.sh               # 启动脚本
├── Dockerfile             # Docker构建文件
├── .env.example           # 环境变量示例
├── data/                  # 数据存储目录
│   ├── *_创业板涨停.csv     # 每日涨停数据
│   └── kline_cache/       # K线数据缓存
└── templates/             # HTML模板
    └── index.html         # 主页面
```

## 腾讯云部署指南

### 方案1：腾讯云轻量级应用服务器部署

1. **购买服务器**
   - 推荐配置：2核4G内存，40GB SSD存储
   - 选择Ubuntu 20.04 LTS系统

2. **环境配置**
   - 安装Python 3.8+
   - 安装Git
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-pip python3-venv git
   ```

3. **部署应用**
   ```bash
   # 克隆项目
   git clone <your-repo-url> cyzt
   cd cyzt
   
   # 创建虚拟环境
   python3 -m venv venv
   source venv/bin/activate
   
   # 安装依赖
   pip install -r requirements.txt
   
   # 配置环境变量
   cp .env.example .env
   # 编辑.env文件配置相关参数
   
   # 给启动脚本添加执行权限
   chmod +x start.sh
   
   # 启动应用
   ./start.sh
   ```

4. **使用PM2管理进程**
   ```bash
   # 安装PM2
   npm install pm2 -g
   
   # 启动应用
   pm2 start "python app.py" --name "cyzt-app"
   
   # 设置开机自启
   pm2 startup
   pm2 save
   ```

### 方案2：腾讯云容器服务TKE部署

1. **准备镜像**
   ```bash
   # 构建Docker镜像
   docker build -t cyzt-app:latest .
   
   # 推送镜像到腾讯云容器镜像服务
   docker tag cyzt-app:latest ccr.ccs.tencentyun.com/your-namespace/cyzt-app:latest
   docker push ccr.ccs.tencentyun.com/your-namespace/cyzt-app:latest
   ```

2. **创建TKE集群**
   - 登录腾讯云控制台，选择容器服务TKE
   - 创建集群并配置节点（推荐2核4G配置）

3. **部署应用**
   - 创建工作负载，使用构建的镜像
   - 配置挂载卷用于数据持久化
   - 暴露服务端口（默认5000）

### 方案3：腾讯云函数SCF + 云开发CloudBase

1. **配置CloudBase环境**
   - 创建CloudBase环境
   - 配置云存储用于存储CSV数据文件

2. **部署Flask应用到CloudBase**
   ```bash
   # 安装CloudBase CLI
   npm install -g @cloudbase/cli
   
   # 登录并部署
   tcb login
   tcb deploy
   ```

3. **配置定时触发**
   - 创建SCF云函数执行数据更新
   - 设置定时触发器（交易日15:30后）

## 数据更新功能说明

### 自动更新
- 应用启动后会自动启动后台线程
- 在每个交易日15:30后自动执行数据更新
- 更新过程在后台运行，不影响Web应用正常访问

### 手动触发
通过API接口手动触发数据更新：
```bash
curl -X POST http://your-domain/api/update/start
```

查看更新状态：
```bash
curl http://your-domain/api/update/status
```

## 数据存储空间配置

- 确保服务器有足够的存储空间用于数据文件
- 创业板涨停数据：每日约1-2KB
- K线缓存数据：每只股票约100KB-1MB
- 建议预留至少1GB空间，以支持历史数据存储

## 性能优化建议

1. **数据目录挂载**
   - 对于大规模部署，建议将数据目录挂载到单独的数据盘
   - 腾讯云可以使用云硬盘或对象存储COS

2. **定期清理旧数据**
   - 可以添加数据清理脚本，定期清理3个月前的历史数据
   - 保留核心数据用于分析

3. **使用CDN加速**
   - 为静态资源配置腾讯云CDN加速
   - 提高访问速度和用户体验

## 故障排查

### 常见问题

1. **数据更新失败**
   - 检查网络连接和API访问权限
   - 查看日志文件中的错误信息
   - 确保pytdx和baostock库配置正确

2. **内存占用过高**
   - 调整批量处理大小
   - 增加服务器内存配置

3. **性能问题**
   - 启用Flask应用的缓存机制
   - 优化K线数据查询逻辑

## 许可证

MIT