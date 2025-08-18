<div align="center">


# Timeseries-MCP

English| [中文](./README_CN.md)

<strong> MCP server supporting multiple time series analysis models </strong>

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](htttps://github.com/kongqi404/Timeseries-MCP/blob/main/LICENSE)


</div>

> [!Note]
>
> ## Under Development 

## Introduction
Timeseries-MCP is a server that supports multiple time series analysis models, including forecasting, anomaly detection, and classification. It is designed to be flexible and extensible, allowing users to easily add new models and functionalities.

## Features
- Support for multiple time series analysis models
- Easy to use and extend
- Built-in minio for data storage


## Supported Models
### Forecasting (WIP)
- ✅ NLinear
- [x] PatchTST
- [x] iTransformer
- [x] Autoformer
- [x] Informer

### Anomaly Detection (WIP)
- [x] NLinear
- [x] PatchTST
- [x] iTransformer

### Classification (WIP)
- [x] NLinear
- [x] PatchTST
- [x] iTransformer

## Installation
### Install from Source

environment requirements: [uv](https://docs.astral.sh/uv/),[podman](https://podman.io/),[git](https://git-scm.com/)

```bash
podman run -p 9000:9000 -p 9001:9001 quay.io/minio/minio server /data --console-address ":9001"

#upload your dataset to minio

git clone https://github.com/kongqi404/Timeseries-MCP.git

cd Timeseries-MCP

# create .env file

uvx . http [--port PORT] [--host HOST] [--env_file FILE_PATH] # http
uvx . stdio [--env_file FILE_PATH] # stdio

```

### Install from Container (WIP)
