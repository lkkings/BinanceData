# =====================================================================
# Binance 实时市场数据服务镜像（多阶段构建，使用国内镜像源）
#
# Mirrors:
#   - Docker base:  docker.m.daocloud.io（DaoCloud Docker Hub 镜像）
#   - pip:          mirrors.tuna.tsinghua.edu.cn（清华 PyPI 镜像）
#   - apt:          mirrors.aliyun.com（阿里云 Debian 镜像）
# =====================================================================

# ---------- 第一阶段：构建依赖 ----------
FROM docker.m.daocloud.io/library/python:3.12-slim AS builder

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# 切换 apt 源到阿里云
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements-runtime.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements-runtime.txt

# ---------- 第二阶段：运行时镜像 ----------
FROM docker.m.daocloud.io/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    TZ=Asia/Shanghai

# 切换 apt 源 + 安装运行时所需的最小工具（curl 用于 healthcheck）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl tzdata \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 复制源码
WORKDIR /app
COPY src/ ./src/
COPY main.py ./

# 数据目录（用于挂载卷）
RUN mkdir -p /app/data/raw

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["python", "main.py"]
