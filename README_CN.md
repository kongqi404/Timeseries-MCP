<div align="center">


# 时间序列多模型控制平台 (Timeseries-MCP)

[English](htttps://github.com/kongqi404/Timeseries-MCP/blob/main/README.md) | 中文

<strong> 支持多种时间序列分析模型的MCP服务器 </strong>

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](htttps://github.com/kongqi404/Timeseries-MCP/blob/main/LICENSE)


</div>

> [!Note]
>
> ## 开发中 

## 项目简介
Timeseries-MCP 是一个支持多种时间序列分析模型的MCP服务器，包含预测、异常检测和分类功能。平台设计为灵活可扩展架构，允许用户便捷地添加新模型和功能模块。

## 核心功能
- 支持多种时间序列分析模型
- 易用且可扩展
- 内置MinIO数据存储

## 支持模型
### 预测功能 (开发中)
- ✅ NLinear
- [x] PatchTST
- [x] iTransformer
- [x] Autoformer
- [x] Informer

### 异常检测 (开发中)
- [x] NLinear
- [x] PatchTST
- [x] iTransformer

### 分类功能 (开发中)
- [x] NLinear
- [x] PatchTST
- [x] iTransformer

## 安装指南
### 源码安装

环境要求: [uv](https://docs.astral.sh/uv/),[podman](https://podman.io/),[git](https://git-scm.com/)

```bash
podman run -p 9000:9000 -p 9001:9001 quay.io/minio/minio server /data --console-address ":9001"

#通过MinIO上传您的数据集

git clone https://github.com/kongqi404/Timeseries-MCP.git

cd Timeseries-MCP

# 创建.env文件

uvx . http [--port PORT] [--host HOST] [--env_file FILE_PATH] # HTTP模式
uvx . stdio [--env_file FILE_PATH] # 标准IO模式

```

### 容器化安装 (开发中)
