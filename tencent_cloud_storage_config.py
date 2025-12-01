#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
腾讯云数据存储配置工具

此脚本用于在腾讯云服务器上配置创业板涨停复盘应用的数据存储空间
包括：
1. 创建数据目录结构
2. 设置适当的权限
3. 配置数据盘挂载（如果使用独立数据盘）
4. 配置对象存储COS（可选）
"""

import os
import sys
import subprocess
import argparse
import json

def print_title(title):
    """打印标题信息"""
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}")

def create_directory_structure(base_dir='./data'):
    """创建必要的目录结构"""
    print(f"\n创建数据目录结构: {base_dir}")
    
    # 创建主数据目录
    os.makedirs(base_dir, exist_ok=True)
    
    # 创建K线缓存目录
    kline_cache_dir = os.path.join(base_dir, 'kline_cache')
    os.makedirs(kline_cache_dir, exist_ok=True)
    
    # 创建日志目录
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"✓ 成功创建目录结构：")
    print(f"  - 主数据目录: {base_dir}")
    print(f"  - K线缓存目录: {kline_cache_dir}")
    print(f"  - 日志目录: {log_dir}")
    
    return {
        'base_dir': base_dir,
        'kline_cache_dir': kline_cache_dir,
        'log_dir': log_dir
    }

def set_permissions(dir_path):
    """设置目录权限"""
    print(f"\n设置目录权限: {dir_path}")
    
    try:
        # 获取当前用户和组
        current_user = subprocess.check_output(['whoami'], universal_newlines=True).strip()
        
        # 设置目录所有者为当前用户
        subprocess.run(['sudo', 'chown', '-R', f'{current_user}:{current_user}', dir_path], check=True)
        
        # 设置目录权限
        subprocess.run(['chmod', '-R', '755', dir_path], check=True)
        
        print(f"✓ 成功设置权限")
        return True
    except Exception as e:
        print(f"✗ 设置权限失败: {str(e)}")
        return False

def create_mount_script(disk_path, mount_point):
    """创建数据盘挂载脚本"""
    print(f"\n创建数据盘挂载脚本")
    
    mount_script = "mount_data_disk.sh"
    with open(mount_script, 'w') as f:
        f.write(f'''
#!/bin/bash

# 腾讯云数据盘挂载脚本
# 使用方法: sudo bash mount_data_disk.sh

echo "开始挂载数据盘..."

# 检查磁盘是否已格式化
disk_formatted=$(sudo file -s {disk_path} | grep -c "ext4")

if [ $disk_formatted -eq 0 ]; then
    echo "磁盘未格式化，开始格式化..."
    sudo mkfs.ext4 {disk_path}
    echo "格式化完成"
else
    echo "磁盘已格式化"
fi

# 创建挂载点
sudo mkdir -p {mount_point}

# 挂载磁盘
sudo mount {disk_path} {mount_point}

# 设置开机自动挂载
echo "设置开机自动挂载..."
sudo cp /etc/fstab /etc/fstab.bak
disk_uuid=$(sudo blkid -s UUID -o value {disk_path})
echo "UUID=$disk_uuid {mount_point} ext4 defaults 0 0" | sudo tee -a /etc/fstab

# 验证挂载
echo "验证挂载..."
df -h {mount_point}

echo "数据盘挂载完成！"
''')
    
    # 给脚本添加执行权限
    os.chmod(mount_script, 0o755)
    
    print(f"✓ 成功创建挂载脚本: {mount_script}")
    print(f"  使用方法: sudo bash {mount_script}")
    return mount_script

def create_cos_config(base_dir='./data', use_cos=False):
    """创建对象存储COS配置文件"""
    if not use_cos:
        print("\n未选择使用对象存储COS")
        return None
    
    print(f"\n创建对象存储COS配置")
    
    cos_config = {
        "enable": True,
        "secret_id": "your-secret-id",
        "secret_key": "your-secret-key",
        "region": "ap-guangzhou",  # 替换为您的COS区域
        "bucket": "your-bucket-name",
        "prefix": "cyzt-data",
        "local_cache_dir": base_dir,
        "sync_interval": 3600  # 同步间隔（秒）
    }
    
    # 保存配置文件
    with open('cos_config.json', 'w') as f:
        json.dump(cos_config, f, indent=4)
    
    print(f"✓ 成功创建COS配置文件: cos_config.json")
    print("  请编辑配置文件，填入您的腾讯云COS凭证信息")
    return cos_config

def create_sync_script(cos_config=None):
    """创建数据同步脚本"""
    if not cos_config:
        print("\n未创建COS同步脚本（未配置COS）")
        return None
    
    print(f"\n创建数据同步脚本")
    
    sync_script = "sync_data_to_cos.py"
    with open(sync_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据同步到腾讯云COS工具

此脚本用于将本地数据目录同步到腾讯云对象存储COS
实现数据备份和跨服务器数据共享
"""

import os
import json
import time
import datetime
import logging
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import threading

def setup_logger():
    """设置日志配置"""
    logger = logging.getLogger('cos_sync')
    logger.setLevel(logging.INFO)
    
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件处理器
    log_file = os.path.join(log_dir, f'cos_sync_{datetime.datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class COSDataSync:
    def __init__(self, config_file='cos_config.json'):
        self.logger = setup_logger()
        self.config = self.load_config(config_file)
        self.client = None
        
        if self.config.get('enable', False):
            self.client = self.init_cos_client()
        else:
            self.logger.warning("COS同步已禁用，请在配置文件中启用")
    
    def load_config(self, config_file):
        """加载配置文件"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f"成功加载配置文件: {config_file}")
            return config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            return {"enable": False}
    
    def init_cos_client(self):
        """初始化COS客户端"""
        try:
            config = CosConfig(
                Region=self.config['region'],
                SecretId=self.config['secret_id'],
                SecretKey=self.config['secret_key']
            )
            client = CosS3Client(config)
            self.logger.info(f"成功初始化COS客户端，区域: {self.config['region']}")
            return client
        except Exception as e:
            self.logger.error(f"初始化COS客户端失败: {str(e)}")
            return None
    
    def upload_file(self, local_file, remote_file):
        """上传单个文件到COS"""
        try:
            response = self.client.put_object_from_local_file(
                Bucket=self.config['bucket'],
                LocalFilePath=local_file,
                Key=remote_file,
                EnableMD5=False
            )
            self.logger.info(f"上传成功: {local_file} -> {remote_file}")
            return True
        except Exception as e:
            self.logger.error(f"上传失败 {local_file}: {str(e)}")
            return False
    
    def sync_directory(self, local_dir, prefix=''):
        """同步本地目录到COS"""
        if not self.client:
            self.logger.error("COS客户端未初始化，同步失败")
            return False
        
        success_count = 0
        fail_count = 0
        
        for root, dirs, files in os.walk(local_dir):
            # 计算相对路径
            relative_path = os.path.relpath(root, local_dir)
            if relative_path == '.':
                relative_path = ''
            
            # 遍历文件并上传
            for file in files:
                local_file_path = os.path.join(root, file)
                
                # 构建远程路径
                if relative_path:
                    remote_key = f"{prefix}/{relative_path}/{file}"
                else:
                    remote_key = f"{prefix}/{file}"
                
                # 上传文件
                if self.upload_file(local_file_path, remote_key):
                    success_count += 1
                else:
                    fail_count += 1
        
        self.logger.info(f"同步完成: 成功 {success_count}, 失败 {fail_count}")
        return fail_count == 0
    
    def run_sync(self):
        """执行同步操作"""
        if not self.config.get('enable', False):
            self.logger.warning("COS同步已禁用")
            return False
        
        self.logger.info("开始数据同步任务")
        
        # 同步K线缓存目录
        kline_cache_dir = os.path.join(self.config['local_cache_dir'], 'kline_cache')
        if os.path.exists(kline_cache_dir):
            self.logger.info(f"同步K线缓存目录: {kline_cache_dir}")
            self.sync_directory(kline_cache_dir, f"{self.config['prefix']}/kline_cache")
        else:
            self.logger.warning(f"K线缓存目录不存在: {kline_cache_dir}")
        
        # 同步涨停数据文件
        self.logger.info(f"同步涨停数据文件: {self.config['local_cache_dir']}")
        
        # 只同步CSV文件
        for file in os.listdir(self.config['local_cache_dir']):
            if file.endswith('.csv'):
                local_file = os.path.join(self.config['local_cache_dir'], file)
                remote_key = f"{self.config['prefix']}/{file}"
                self.upload_file(local_file, remote_key)
        
        self.logger.info("数据同步任务完成")
        return True
    
    def start_auto_sync(self):
        """启动自动同步任务"""
        self.logger.info("启动自动数据同步服务")
        
        while True:
            try:
                self.run_sync()
            except Exception as e:
                self.logger.error(f"自动同步任务异常: {str(e)}")
            
            # 等待下一次同步
            sync_interval = self.config.get('sync_interval', 3600)
            self.logger.info(f"等待 {sync_interval} 秒后进行下一次同步")
            time.sleep(sync_interval)

def main():
    sync_tool = COSDataSync()
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # 启动自动同步服务
        thread = threading.Thread(target=sync_tool.start_auto_sync, daemon=True)
        thread.start()
        sync_tool.logger.info("自动同步服务已启动，按 Ctrl+C 停止")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sync_tool.logger.info("自动同步服务已停止")
    else:
        # 执行单次同步
        sync_tool.run_sync()

if __name__ == "__main__":
    main()
''')
    
    # 给脚本添加执行权限
    os.chmod(sync_script, 0o755)
    
    print(f"✓ 成功创建数据同步脚本: {sync_script}")
    print(f"  单次同步: python {sync_script}")
    print(f"  自动同步: python {sync_script} --auto")
    
    # 添加COS SDK依赖到requirements.txt
    try:
        with open('requirements.txt', 'a') as f:
            f.write('cos-python-sdk-v5==1.9.2\n')
        print("✓ 已添加COS SDK依赖到requirements.txt")
    except Exception as e:
        print(f"✗ 添加COS SDK依赖失败: {str(e)}")
    
    return sync_script

def main():
    parser = argparse.ArgumentParser(description='腾讯云数据存储配置工具')
    parser.add_argument('--base-dir', default='./data', help='数据存储基础目录')
    parser.add_argument('--setup-disk', action='store_true', help='设置数据盘挂载')
    parser.add_argument('--disk-path', default='/dev/vdb', help='数据盘路径')
    parser.add_argument('--use-cos', action='store_true', help='使用腾讯云对象存储COS')
    parser.add_argument('--set-permissions', action='store_true', help='设置目录权限')
    
    args = parser.parse_args()
    
    print_title("腾讯云数据存储配置工具")
    
    # 创建目录结构
    dirs = create_directory_structure(args.base_dir)
    
    # 设置权限
    if args.set_permissions:
        set_permissions(args.base_dir)
    else:
        print("\n提示：如果遇到权限问题，可以使用 --set-permissions 参数设置目录权限")
    
    # 设置数据盘挂载
    if args.setup_disk:
        mount_script = create_mount_script(args.disk_path, args.base_dir)
    else:
        print("\n提示：如果需要挂载独立数据盘，可以使用 --setup-disk 参数")
    
    # 配置COS
    cos_config = create_cos_config(args.base_dir, args.use_cos)
    if args.use_cos:
        create_sync_script(cos_config)
    else:
        print("\n提示：如果需要使用对象存储COS备份数据，可以使用 --use-cos 参数")
    
    print_title("配置完成")
    print(f"\n推荐配置：")
    print(f"1. 腾讯云轻量级应用服务器：建议选择40GB SSD存储")
    print(f"2. 腾讯云CVM：建议挂载独立数据盘（≥50GB）")
    print(f"3. 大规模部署：使用对象存储COS + 本地缓存")
    print(f"\n数据存储路径：{dirs['base_dir']}")
    print(f"K线缓存路径：{dirs['kline_cache_dir']}")
    print(f"日志存储路径：{dirs['log_dir']}")
    print(f"\n建议监控存储空间使用情况，确保有足够空间存储历史数据")

if __name__ == "__main__":
    main()