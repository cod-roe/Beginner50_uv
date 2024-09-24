FROM python:3.12

# パッケージのインストール
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        tzdata sudo\
    && rm -rf /var/lib/apt/lists/*

# モジュールのインストール
WORKDIR /opt
RUN pip install -U pip \
    && pip install uv

# COPY .python-version pyproject.toml uv.lock ${PWD}
# RUN uv python pin "$(cat .python-version)" 
# 環境変数の定義
ENV TZ="Asia/Tokyo"