import os
import requests
import zipfile
import pandas as pd
from datetime import datetime, timedelta

BASE_URL = "https://data.binance.vision/data/spot/daily/trades"


def download_day(symbol, date_str, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{symbol}-trades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    local_zip = os.path.join(save_dir, filename)

    if os.path.exists(local_zip):
        print(f"Already downloaded: {filename}")
        return local_zip

    print(f"Downloading {filename}...")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print(f"Failed: {url}")
        return None

    with open(local_zip, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_zip


def unzip_file(zip_path, extract_dir="data"):
    output_path = zip_path.replace('.zip', '.csv')
    if os.path.exists(output_path):
        return output_path
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return output_path


def load_and_resample(csv_path):
    
    df = pd.read_csv(csv_path, header=None)
    
    df.columns = ["trade_id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "ignore"]
    
    # 时间处理（你这里是对的：us）
    df['time'] = pd.to_datetime(df['time'], unit='us')
    df = df.sort_values('time')
    df.set_index('time', inplace=True)

    # ===== 1秒K线 =====
    ohlc = df['price'].resample('1s').ohlc()

    # ===== 成交量（非常重要）=====
    volume = df['qty'].resample('1s').sum()

    # ===== 合并 =====
    ohlc['volume'] = volume

    # ===== 处理空值 =====
    ohlc['close'] = ohlc['close'].ffill()
    ohlc['open'] = ohlc['open'].fillna(ohlc['close'])
    ohlc['high'] = ohlc['high'].fillna(ohlc['close'])
    ohlc['low']  = ohlc['low'].fillna(ohlc['close'])
    ohlc['volume'] = ohlc['volume'].fillna(0)

    ohlc = ohlc.dropna()
    return ohlc

def analyze_threshold(ohlc, threshold=0.001):
    prices = ohlc['close'].to_numpy()  # ✅ 强制转 numpy
    
    up, down = 0, 0
    last_extreme = prices[0]
    
    for p in prices[1:]:
        change = (p - last_extreme) / last_extreme
        
        if  change >= threshold:
            up += 1
            last_extreme = p
            
        elif change <= -threshold:
            down += 1
            last_extreme = p
    
    return up, down

def run(symbol, start_date, end_date, threshold=0.001):
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    results = []

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"Processing {date_str}")

        zip_path = download_day(symbol, date_str)
        if not zip_path:
            current += timedelta(days=1)
            continue

        csv_path = unzip_file(zip_path)
        ohlc = load_and_resample(csv_path)

        up, down = analyze_threshold(ohlc, threshold)

        results.append({
            "date": date_str,
            "up_moves": int(up),
            "down_moves": int(down)
        })

        current += timedelta(days=1)

    result_df = pd.DataFrame(results)
    print("\nSummary:")
    print(result_df)

    print("\nDistribution:")
    print(result_df.describe())

    return result_df


if __name__ == "__main__":
    # 示例参数
    symbol = "BTCUSDT"
    start_date = "2026-05-01"
    end_date = "2026-05-13"
    threshold = 0.001  # 0.1%

    run(symbol, start_date, end_date, threshold)
