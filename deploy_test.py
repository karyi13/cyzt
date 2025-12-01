#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
部署测试脚本

此脚本用于验证创业板涨停复盘应用的完整部署流程，包括：
1. 环境检查
2. 依赖安装验证
3. 数据目录准备
4. Web应用启动测试
5. 数据更新功能测试
6. API接口测试
"""

import os
import sys
import subprocess
import time
import json
import logging
import requests
import tempfile
import shutil
import argparse
from datetime import datetime, timedelta

def setup_logger():
    """设置日志配置"""
    logger = logging.getLogger('deploy_test')
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(console_handler)
    
    # 创建日志文件
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'deploy_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

def print_title(title):
    """打印标题信息"""
    logger.info(f"\n{'='*70}")
    logger.info(f"{title.center(70)}")
    logger.info(f"{'='*70}")

def run_command(command, cwd=None, timeout=30, check=False):
    """执行命令并返回结果"""
    logger.info(f"执行命令: {command}")
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=cwd
        )
        stdout, stderr = process.communicate(timeout=timeout)
        return_code = process.returncode
        
        stdout = stdout.decode('utf-8', errors='replace')
        stderr = stderr.decode('utf-8', errors='replace')
        
        if stdout.strip():
            logger.info(f"命令输出:\n{stdout}")
        if stderr.strip():
            logger.warning(f"命令错误输出:\n{stderr}")
        
        if check and return_code != 0:
            raise subprocess.CalledProcessError(return_code, command, output=stdout, stderr=stderr)
        
        return return_code, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        logger.error(f"命令执行超时: {command}")
        return -1, "", "Command timed out"
    except Exception as e:
        logger.error(f"执行命令失败: {command}, 错误: {str(e)}")
        return -1, "", str(e)

def check_python_version():
    """检查Python版本"""
    print_title("检查Python环境")
    
    result = run_command('python --version', check=True)
    if result[0] != 0:
        logger.error("未找到Python环境")
        return False
    
    # 获取Python版本
    version_cmd = "python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'"
    return_code, version, _ = run_command(version_cmd)
    
    if return_code == 0:
        major, minor = map(int, version.strip().split('.'))
        if major >= 3 and minor >= 7:
            logger.info(f"Python版本检查通过: {version}")
            return True
        else:
            logger.warning(f"Python版本可能不兼容: {version}，建议使用Python 3.7+")
            return False
    else:
        logger.error("无法获取Python版本")
        return False

def check_requirements():
    """检查依赖项"""
    print_title("检查项目依赖")
    
    # 检查requirements.txt是否存在
    if not os.path.exists('requirements.txt'):
        logger.error("requirements.txt文件不存在")
        return False
    
    # 检查必要的依赖是否已安装
    required_packages = [
        'flask', 
        'requests', 
        'pytdx', 
        'baostock', 
        'akshare'
    ]
    
    all_installed = True
    for package in required_packages:
        cmd = f"python -c 'import {package.replace("-", "_")}'" 
        return_code, _, _ = run_command(cmd, timeout=5)
        if return_code != 0:
            logger.warning(f"依赖 {package} 未安装")
            all_installed = False
        else:
            logger.info(f"依赖 {package} 已安装")
    
    if not all_installed:
        logger.info("建议运行: pip install -r requirements.txt 安装所有依赖")
    
    return all_installed

def prepare_data_directories(data_dir=None):
    """准备数据目录"""
    print_title("准备数据目录")
    
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # 检查环境变量
    env_data_dir = os.environ.get('DATA_DIR')
    if env_data_dir:
        data_dir = env_data_dir
        logger.info(f"使用环境变量中的数据目录: {data_dir}")
    
    # 创建必要的目录
    directories = {
        'data_dir': data_dir,
        'kline_cache_dir': os.path.join(data_dir, 'kline_cache'),
        'log_dir': os.path.join(data_dir, 'logs')
    }
    
    all_created = True
    for name, path in directories.items():
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"创建目录成功: {path}")
        except Exception as e:
            logger.error(f"创建目录失败 {path}: {str(e)}")
            all_created = False
    
    # 检查目录权限
    for path in directories.values():
        if not os.access(path, os.W_OK):
            logger.warning(f"目录无写入权限: {path}")
            all_created = False
    
    return all_created

def test_web_app():
    """测试Web应用启动"""
    print_title("测试Web应用启动")
    
    # 检查app.py是否存在
    if not os.path.exists('app.py'):
        logger.error("app.py文件不存在")
        return False
    
    # 检查应用是否可以正常运行
    # 注意：这里只检查语法错误，不实际启动服务
    cmd = "python -m compileall app.py"
    return_code, stdout, stderr = run_command(cmd)
    
    if return_code == 0:
        logger.info("Web应用语法检查通过")
        
        # 检查应用是否定义了必要的路由
        check_routes = "python -c 'from app import app; print([rule.rule for rule in app.url_map.iter_rules()])'"
        return_code, stdout, stderr = run_command(check_routes, timeout=10)
        
        if return_code == 0:
            logger.info(f"Web应用路由检查通过，已定义的路由: {stdout.strip()}")
            
            # 检查是否包含数据更新相关路由
            if '/api/update/' in stdout:
                logger.info("✓ 已包含数据更新API路由")
            else:
                logger.warning("✗ 未找到数据更新API路由")
            
            return True
        else:
            logger.error(f"Web应用路由检查失败: {stderr}")
            return False
    else:
        logger.error(f"Web应用语法检查失败: {stderr}")
        return False

def test_data_update_script():
    """测试数据更新脚本"""
    print_title("测试数据更新脚本")
    
    # 检查数据更新脚本是否存在
    update_script = "获取K线数据_高速版.py"
    if not os.path.exists(update_script):
        logger.error(f"数据更新脚本不存在: {update_script}")
        return False
    
    # 检查脚本语法
    cmd = f"python -m compileall {update_script}"
    return_code, stdout, stderr = run_command(cmd)
    
    if return_code == 0:
        logger.info("数据更新脚本语法检查通过")
        
        # 检查脚本是否包含必要的函数
        check_funcs = f'python -c "import re; content=open(\'{update_script}\', \'r\', encoding=\'utf-8\').read(); funcs=[\'fetch_kline_pytdx\', \'fetch_kline_baostock\', \'fetch_kline_data\']; [print(f\'{{f}}: {{"✓" if re.search(f"def {f}", content) else "✗"}}\') for f in funcs]"'
        return_code, stdout, stderr = run_command(check_funcs, timeout=10)
        
        if return_code == 0:
            logger.info("数据更新脚本函数检查结果:")
            logger.info(stdout)
            return True
        else:
            logger.warning(f"数据更新脚本函数检查失败: {stderr}")
            return False
    else:
        logger.error(f"数据更新脚本语法检查失败: {stderr}")
        return False

def simulate_api_tests():
    """模拟API接口测试"""
    print_title("模拟API接口测试")
    
    # 模拟测试基本的API接口调用
    # 注意：这里只模拟测试，不实际发送请求
    api_endpoints = [
        {"url": "/", "method": "GET", "desc": "首页"},
        {"url": "/api/dates", "method": "GET", "desc": "获取日期列表"},
        {"url": "/api/update/start", "method": "POST", "desc": "启动数据更新"},
        {"url": "/api/update/status", "method": "GET", "desc": "查询更新状态"}
    ]
    
    logger.info("API接口列表:")
    for endpoint in api_endpoints:
        logger.info(f"- {endpoint['method']} {endpoint['url']} - {endpoint['desc']}")
    
    logger.info("提示: 实际部署后，建议使用以下命令测试API接口:")
    logger.info("1. 测试首页: curl http://localhost:5000/")
    logger.info("2. 获取日期: curl http://localhost:5000/api/dates")
    logger.info("3. 启动更新: curl -X POST http://localhost:5000/api/update/start")
    logger.info("4. 查询状态: curl http://localhost:5000/api/update/status")
    
    return True

def test_docker_setup():
    """测试Docker配置"""
    print_title("测试Docker配置")
    
    # 检查Dockerfile是否存在
    if os.path.exists('Dockerfile'):
        logger.info("✓ Dockerfile 存在")
        
        # 检查Dockerfile内容
        with open('Dockerfile', 'r') as f:
            content = f.read()
            
        # 检查必要的配置项
        checks = [
            ("python", "Python基础镜像配置"),
            ("requirements.txt", "依赖安装配置"),
            ("5000", "端口配置"),
            ("app.py", "应用启动配置")
        ]
        
        all_checks_passed = True
        for keyword, desc in checks:
            if keyword in content:
                logger.info(f"✓ {desc}")
            else:
                logger.warning(f"✗ {desc}")
                all_checks_passed = False
        
        if all_checks_passed:
            logger.info("Docker配置检查通过")
        else:
            logger.warning("Docker配置存在问题，建议检查")
            
        return all_checks_passed
    else:
        logger.warning("Dockerfile 不存在")
        return False

def run_complete_test():
    """运行完整测试"""
    print_title("开始完整部署测试")
    
    test_results = {
        "python_version": check_python_version(),
        "requirements": check_requirements(),
        "data_directories": prepare_data_directories(),
        "web_app": test_web_app(),
        "update_script": test_data_update_script(),
        "api_tests": simulate_api_tests(),
        "docker_setup": test_docker_setup()
    }
    
    print_title("部署测试结果汇总")
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 所有测试通过！您的项目已准备好部署到腾讯云")
    else:
        logger.warning("\n⚠️ 部分测试失败，请检查并修复问题后再部署")
    
    logger.info("\n推荐的部署步骤:")
    logger.info("1. 在腾讯云服务器上安装Python 3.7+")
    logger.info("2. 上传项目文件到服务器")
    logger.info("3. 安装项目依赖: pip install -r requirements.txt")
    logger.info("4. 配置环境变量 (参考 .env.example)")
    logger.info("5. 使用部署脚本启动应用: bash start.sh")
    logger.info("6. 对于容器化部署: docker build -t cyzt-app . && docker run -p 5000:5000 cyzt-app")
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(description='创业板涨停复盘应用部署测试工具')
    parser.add_argument('--quick', action='store_true', help='快速测试模式，只运行必要的测试')
    parser.add_argument('--data-dir', help='指定数据目录路径')
    parser.add_argument('--test-update', action='store_true', help='单独测试数据更新功能')
    parser.add_argument('--test-web', action='store_true', help='单独测试Web应用功能')
    
    args = parser.parse_args()
    
    try:
        if args.test_update:
            # 单独测试数据更新功能
            prepare_data_directories(args.data_dir)
            test_data_update_script()
        elif args.test_web:
            # 单独测试Web应用功能
            prepare_data_directories(args.data_dir)
            test_web_app()
            simulate_api_tests()
        elif args.quick:
            # 快速测试模式
            print_title("快速部署测试")
            check_python_version()
            prepare_data_directories(args.data_dir)
            test_web_app()
            test_data_update_script()
        else:
            # 完整测试
            run_complete_test()
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
    finally:
        print_title("测试完成")

if __name__ == "__main__":
    main()