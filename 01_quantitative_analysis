"""
Tesla (TSLA) 5年股票数据分析 / Tesla 5-Year Stock Data Analysis
对比基准：S&P 500 (^GSPC)
"""

import sys
# 修复 Windows 终端中文乱码：强制 stdout/stderr 使用 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import warnings
import time
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')   # 必须在其他 matplotlib 导入前设置，确保 Windows 无界面下能保存 PNG
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='muted')

# ── 直接调用 Yahoo Finance v8 Chart API（绕过 yfinance 内部限流问题）──
_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    )
}

def fetch_ohlcv(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """从 Yahoo Finance v8 Chart API 获取日K数据，返回 OHLCV DataFrame。"""
    p1 = int(start.timestamp())
    p2 = int(end.timestamp())
    url = (
        f'https://query2.finance.yahoo.com/v8/finance/chart/{ticker}'
        f'?interval=1d&period1={p1}&period2={p2}&events=splits,dividends'
    )
    for attempt in range(3):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            q = result['indicators']['quote'][0]
            adj = result['indicators'].get('adjclose', [{}])[0].get('adjclose', q['close'])
            df = pd.DataFrame({
                'Open':   q['open'],
                'High':   q['high'],
                'Low':    q['low'],
                'Close':  adj,       # 使用复权收盘价
                'Volume': q['volume'],
            }, index=pd.to_datetime(timestamps, unit='s', utc=True).tz_convert('America/New_York').normalize().tz_localize(None))
            df.index.name = 'Date'
            return df.dropna()
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                raise RuntimeError(f"无法获取 {ticker} 数据: {e}")

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: 数据获取 / Data Acquisition
# ============================================================
print("=" * 60)
print("STEP 1: 数据获取 / Data Acquisition")
print("  说明：从 Yahoo Finance 下载特斯拉近5年的日K线数据，")
print("        同时下载 S&P 500 作为对比基准，是后续所有计算的基础。")
print("=" * 60)

END_DATE   = datetime.today().strftime('%Y-%m-%d')
START_DATE = (datetime.today() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

print(f"  下载范围：{START_DATE} → {END_DATE}")
start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
end_dt   = datetime.strptime(END_DATE,   '%Y-%m-%d')

print("  正在获取 TSLA 数据...")
tsla_raw  = fetch_ohlcv('TSLA',  start_dt, end_dt)
print("  正在获取 S&P 500 数据...")
sp500_raw = fetch_ohlcv('%5EGSPC', start_dt, end_dt)   # ^GSPC URL编码

# 只保留收盘价，按日期内连接对齐
close = pd.DataFrame({
    'TSLA':  tsla_raw['Close'],
    'SP500': sp500_raw['Close'],
}).dropna()   # 删除任一为空的行，确保两列完全对齐

tsla_close  = close['TSLA']
sp500_close = close['SP500']

print(f"  有效交易日数：{len(close)}")
print(f"  实际日期范围：{close.index[0].date()} → {close.index[-1].date()}")
print(f"\n  前3行数据：")
print(close.head(3).to_string())
print(f"\n  基本统计摘要 (describe)：")
print(close.describe().to_string())
print()

# ============================================================
# STEP 2: 收益率计算 / Return Rate Calculation
# ============================================================
print("=" * 60)
print("STEP 2: 收益率计算 / Return Rate Calculation")
print("  说明：收益率衡量每天/每年赚了多少，累计收益率展示从起点")
print("        到今天的总增值倍数；对齐起点使 TSLA 与大盘可直接比较。")
print("=" * 60)

tsla_daily  = tsla_close.pct_change()
sp500_daily = sp500_close.pct_change()
tsla_log    = np.log(tsla_close  / tsla_close.shift(1))
sp500_log   = np.log(sp500_close / sp500_close.shift(1))

# dropna 统一在此调用一次，后续所有序列无 NaN
tsla_daily  = tsla_daily.dropna()
sp500_daily = sp500_daily.dropna()
tsla_log    = tsla_log.dropna()
sp500_log   = sp500_log.dropna()

# 对齐起点的累计收益率：两条线都从 0% 出发
tsla_cum  = (1 + tsla_daily).cumprod()
sp500_cum = (1 + sp500_daily).cumprod()
tsla_cum  = tsla_cum  / tsla_cum.iloc[0]  - 1
sp500_cum = sp500_cum / sp500_cum.iloc[0] - 1

# 各年度收益率
year_idx = tsla_daily.index.year
tsla_annual_return  = tsla_daily.groupby(year_idx).apply(lambda x: (1 + x).prod() - 1)
sp500_annual_return = sp500_daily.groupby(year_idx).apply(lambda x: (1 + x).prod() - 1)

# 胜率（日涨幅 > 0 的比例）
tsla_win_rate  = (tsla_daily  > 0).sum() / len(tsla_daily)
sp500_win_rate = (sp500_daily > 0).sum() / len(sp500_daily)

print(f"  TSLA  日收益率：均值={tsla_daily.mean()*100:.4f}%  "
      f"标准差={tsla_daily.std()*100:.4f}%  "
      f"最大={tsla_daily.max()*100:.2f}%  最小={tsla_daily.min()*100:.2f}%")
print(f"  SP500 日收益率：均值={sp500_daily.mean()*100:.4f}%  "
      f"标准差={sp500_daily.std()*100:.4f}%  "
      f"最大={sp500_daily.max()*100:.2f}%  最小={sp500_daily.min()*100:.2f}%")

print(f"\n  日胜率对比：TSLA {tsla_win_rate*100:.2f}%  |  S&P 500 {sp500_win_rate*100:.2f}%")

print(f"\n  各年度收益率对比：")
annual_df = pd.DataFrame({'TSLA': tsla_annual_return * 100,
                          'SP500': sp500_annual_return * 100}).round(2)
print(annual_df.to_string())
print()

# ============================================================
# STEP 3: 波动率计算 / Volatility Calculation
# ============================================================
print("=" * 60)
print("STEP 3: 波动率计算 / Volatility Calculation")
print("  说明：波动率衡量价格波动的剧烈程度，年化波动率越高代表")
print("        风险越大；Sharpe 比率衡量每单位风险所获得的回报。")
print("=" * 60)

TRADING_DAYS = 252

# CAGR：基于实际交易日数，两支股票使用同一 years 值确保口径一致
years      = len(tsla_daily) / TRADING_DAYS
tsla_cagr  = (tsla_close.iloc[-1] / tsla_close.iloc[0]) ** (1 / years) - 1
sp500_cagr = (sp500_close.iloc[-1] / sp500_close.iloc[0]) ** (1 / years) - 1

# 年化波动率：日对数收益标准差 × √252
tsla_ann_vol  = tsla_log.std()  * np.sqrt(TRADING_DAYS)
sp500_ann_vol = sp500_log.std() * np.sqrt(TRADING_DAYS)

# 30日滚动年化波动率（保留前29行 NaN，维持时序完整性）
rolling_vol_30d = tsla_log.rolling(30).std() * np.sqrt(TRADING_DAYS)

# 各年度波动率（仅 TSLA）
tsla_annual_vol = (tsla_log.groupby(year_idx).std() * np.sqrt(TRADING_DAYS))

# 标准年化 Sharpe（RF=0）：日对数收益均值/标准差 × √252
# 即 年化超额收益 / 年化波动率，RF 取 0
sharpe = (tsla_log.mean() / tsla_log.std()) * np.sqrt(TRADING_DAYS)

# 最大回撤序列
cum_wealth      = (1 + tsla_daily).cumprod()
rolling_peak    = cum_wealth.cummax()
drawdown_series = cum_wealth / rolling_peak - 1   # 始终 ≤ 0
max_drawdown    = drawdown_series.min()
min_date        = drawdown_series.idxmin()
min_value       = drawdown_series.min()

print(f"  实际交易日数：{len(tsla_daily)} 天，等效年数：{years:.2f} 年")
print(f"\n  CAGR 对比（基于相同实际交易日）：")
print(f"    TSLA  CAGR：{tsla_cagr*100:.2f}%")
print(f"    SP500 CAGR：{sp500_cagr*100:.2f}%")
print(f"\n  年化波动率对比：")
print(f"    TSLA  年化波动率：{tsla_ann_vol*100:.2f}%")
print(f"    SP500 年化波动率：{sp500_ann_vol*100:.2f}%")
print(f"    TSLA 波动率是大盘的 {tsla_ann_vol/sp500_ann_vol:.1f} 倍")
print(f"\n  标准年化 Sharpe 比率（RF=0）：{sharpe:.4f}")
print(f"  30日滚动波动率（最新值）：{rolling_vol_30d.dropna().iloc[-1]*100:.2f}%")
print(f"\n  最大回撤：{max_drawdown*100:.2f}%  发生日期：{min_date.date()}")
print()

# ============================================================
# STEP 4: 可视化 / Visualization
# ============================================================
print("=" * 60)
print("STEP 4: 可视化 / Visualization")
print("  说明：用图形直观呈现价格走势、收益分布、累计收益对比、")
print("        波动率变化、最大回撤及年度收益，帮助快速把握整体状况。")
print("=" * 60)

# ── 图1：股价历史 + 成交量 ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
sma50  = tsla_close.rolling(50).mean()
sma200 = tsla_close.rolling(200).mean()

ax1.plot(tsla_close.index, tsla_close, label='Close', lw=1.2, color='#1f77b4')
ax1.plot(sma50.index,  sma50,  label='SMA-50',  lw=1, color='orange', ls='--')
ax1.plot(sma200.index, sma200, label='SMA-200', lw=1, color='red',    ls='--')
ax1.set_title('TSLA 5-Year Price History\n特斯拉5年股价历史（含成交量）',
              fontsize=14, fontweight='bold')
ax1.set_ylabel('Price (USD)')
ax1.legend(loc='upper left')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

volume = tsla_raw['Volume'].reindex(tsla_close.index)
ax2.bar(tsla_close.index, volume, color='steelblue', alpha=0.5, width=1)
ax2.set_ylabel('Volume')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '01_price_history.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  已保存：01_price_history.png")

# ── 图2：日收益率分布 ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
clean = tsla_daily.dropna()
ax.hist(clean, bins=100, density=True, alpha=0.6, color='steelblue', label='Daily Returns')
clean.plot.kde(ax=ax, color='darkblue', lw=2, label='KDE')
ax.axvline(0,           color='black', lw=1,   ls='--', label='Zero')
ax.axvline(clean.mean(),color='red',   lw=1.5, ls='-',
           label=f'Mean={clean.mean()*100:.3f}%')
skew = clean.skew()
kurt = clean.kurt()
ax.text(0.97, 0.95,
        f'Skewness: {skew:.3f}\nKurtosis: {kurt:.3f}',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_title('TSLA Daily Return Distribution\n特斯拉日收益率分布', fontsize=14, fontweight='bold')
ax.set_xlabel('Daily Return')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '02_daily_returns.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  已保存：02_daily_returns.png")

# ── 图3：TSLA vs S&P 500 累计收益对比（起点对齐为 0%）───────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(tsla_cum.index,  tsla_cum  * 100, color='#1f77b4', lw=1.5, label='TSLA')
ax.plot(sp500_cum.index, sp500_cum * 100, color='#ff7f0e', lw=1.5, label='S&P 500')
ax.fill_between(tsla_cum.index, tsla_cum * 100, sp500_cum * 100,
                where=(tsla_cum >= sp500_cum), alpha=0.1, color='blue',  label='TSLA领先')
ax.fill_between(tsla_cum.index, tsla_cum * 100, sp500_cum * 100,
                where=(tsla_cum <  sp500_cum), alpha=0.1, color='orange', label='SP500领先')
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.set_title('TSLA vs S&P 500 Cumulative Return (Aligned Start)\n累计收益率对比（起点统一归零）',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return (%)')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '03_cumulative_returns.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  已保存：03_cumulative_returns.png")

# ── 图4：TSLA 30日滚动年化波动率 ──────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(rolling_vol_30d.index, rolling_vol_30d * 100,
        color='darkorange', lw=1.2, label='30-Day Rolling Vol (Ann.)')
ax.axhline(tsla_ann_vol * 100, color='red',  lw=1.5, ls='--',
           label=f'TSLA 5Y Ann. Vol: {tsla_ann_vol*100:.1f}%')
ax.axhline(sp500_ann_vol * 100, color='green', lw=1.5, ls=':',
           label=f'SP500 5Y Ann. Vol: {sp500_ann_vol*100:.1f}%')
ax.fill_between(rolling_vol_30d.index, rolling_vol_30d * 100, alpha=0.2, color='orange')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.set_title('TSLA 30-Day Rolling Annualised Volatility\n特斯拉30日滚动年化波动率（附S&P 500基准线）',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Annualised Volatility (%)')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '04_volatility.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  已保存：04_volatility.png")

# ── 图5：各年度收益率柱状图 ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ca02c' if v >= 0 else '#d62728' for v in tsla_annual_return.values]
bars = ax.bar(tsla_annual_return.index.astype(str),
              tsla_annual_return.values * 100,
              color=colors, edgecolor='black', lw=0.5)
for bar, val in zip(bars, tsla_annual_return.values):
    offset = 1.5 if val >= 0 else -3.5
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f'{val*100:.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.axhline(0, color='black', lw=0.8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.set_title('TSLA Annual Returns by Year\n特斯拉各年度收益率', fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Annual Return (%)')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '05_annual_returns.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  已保存：05_annual_returns.png")

# ── 图6：最大回撤曲线（面积图 + 精确标注）──────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(drawdown_series.index, drawdown_series * 100, 0,
                color='red', alpha=0.4, label='Drawdown')
ax.plot(drawdown_series.index, drawdown_series * 100, color='darkred', lw=0.8)
ax.axhline(0, color='black', lw=0.8, ls='--')
# 精确标注最低点：箭头 + 数值 + 日期
ax.annotate(
    f'{min_value*100:.1f}%\n({min_date.strftime("%Y-%m-%d")})',
    xy=(min_date, min_value * 100),
    xytext=(40, 40), textcoords='offset points',
    fontsize=10, fontweight='bold', color='darkred',
    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5)
)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.set_title('TSLA Maximum Drawdown\n特斯拉最大回撤曲线', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '06_max_drawdown.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  已保存：06_max_drawdown.png")

# ── 图7：综合仪表板（2×3 六格）─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('TSLA 5-Year Stock Analysis Dashboard\n特斯拉5年股票分析综合仪表板',
             fontsize=16, fontweight='bold')

# [0,0] 股价
axes[0, 0].plot(tsla_close.index, tsla_close, lw=0.8, color='#1f77b4', label='Close')
axes[0, 0].plot(sma50.index,  sma50,  lw=0.7, color='orange', ls='--', label='SMA50')
axes[0, 0].plot(sma200.index, sma200, lw=0.7, color='red',    ls='--', label='SMA200')
axes[0, 0].set_title('Price History 股价历史')
axes[0, 0].legend(fontsize=7)

# [0,1] 累计收益对比
axes[0, 1].plot(tsla_cum.index,  tsla_cum  * 100, color='#1f77b4', lw=1, label='TSLA')
axes[0, 1].plot(sp500_cum.index, sp500_cum * 100, color='#ff7f0e', lw=1, label='S&P 500')
axes[0, 1].axhline(0, color='black', lw=0.6, ls='--')
axes[0, 1].set_title('Cumulative Return 累计收益率对比')
axes[0, 1].legend(fontsize=7)

# [0,2] 滚动波动率
axes[0, 2].plot(rolling_vol_30d.index, rolling_vol_30d * 100,
                color='darkorange', lw=0.8, label='30D Rolling Vol')
axes[0, 2].axhline(tsla_ann_vol * 100, color='red',   lw=1, ls='--',
                   label=f'TSLA {tsla_ann_vol*100:.0f}%')
axes[0, 2].axhline(sp500_ann_vol * 100, color='green', lw=1, ls=':',
                   label=f'SP500 {sp500_ann_vol*100:.0f}%')
axes[0, 2].set_title('Rolling Volatility 滚动波动率')
axes[0, 2].legend(fontsize=7)

# [1,0] 年度收益
c = ['#2ca02c' if v >= 0 else '#d62728' for v in tsla_annual_return.values]
axes[1, 0].bar(tsla_annual_return.index.astype(str),
               tsla_annual_return.values * 100, color=c, edgecolor='black', lw=0.5)
axes[1, 0].axhline(0, color='black', lw=0.6)
axes[1, 0].set_title('Annual Returns 年度收益率')

# [1,1] 最大回撤
axes[1, 1].fill_between(drawdown_series.index, drawdown_series * 100, 0,
                         color='red', alpha=0.4)
axes[1, 1].plot(drawdown_series.index, drawdown_series * 100,
                color='darkred', lw=0.6)
axes[1, 1].axhline(0, color='black', lw=0.6, ls='--')
axes[1, 1].set_title('Max Drawdown 最大回撤')

# [1,2] 日收益分布
axes[1, 2].hist(tsla_daily, bins=80, density=True, alpha=0.6,
                color='steelblue', label='TSLA')
axes[1, 2].hist(sp500_daily, bins=80, density=True, alpha=0.5,
                color='orange',   label='S&P 500')
axes[1, 2].axvline(0, color='black', lw=0.8, ls='--')
axes[1, 2].set_title('Daily Return Dist. 日收益率分布')
axes[1, 2].legend(fontsize=7)

for ax in axes.flat:
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '07_summary_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  已保存：07_summary_dashboard.png")
print()

# ============================================================
# STEP 5: 分析结论 / Analysis Conclusions
# ============================================================
print("=" * 60)
print("STEP 5: 分析结论 / Analysis Conclusions")
print("  说明：基于以上数据，给出对特斯拉投资价值的判断性结论，")
print("        而非堆砌数字——每条结论都有明确的观点。")
print("=" * 60)

# ── 原始数值块（供参考）───────────────────────────────────
print("\n【原始数值参考】")
print(f"  起始价格   : ${tsla_close.iloc[0]:,.2f}   →   最新价格：${tsla_close.iloc[-1]:,.2f}")
print(f"  5年最高价  : ${tsla_close.max():,.2f} ({tsla_close.idxmax().date()})")
print(f"  5年最低价  : ${tsla_close.min():,.2f} ({tsla_close.idxmin().date()})")
print(f"  TSLA CAGR  : {tsla_cagr*100:.2f}%   |   SP500 CAGR：{sp500_cagr*100:.2f}%")
print(f"  TSLA 年化波动率：{tsla_ann_vol*100:.2f}%   |   SP500 年化波动率：{sp500_ann_vol*100:.2f}%")
print(f"  TSLA 日胜率：{tsla_win_rate*100:.2f}%   |   SP500 日胜率：{sp500_win_rate*100:.2f}%")
print(f"  标准年化 Sharpe（RF=0）：{sharpe:.4f}")
print(f"  最大回撤：{max_drawdown*100:.2f}%（{min_date.date()}）")

# ── 判断性结论 ──────────────────────────────────────────────
cagr_gap = (tsla_cagr - sp500_cagr) * 100
outperform = "跑赢" if tsla_cagr > sp500_cagr else "跑输"

vol_ratio = tsla_ann_vol / sp500_ann_vol
if tsla_ann_vol > 0.6:
    vol_level = "极高"
elif tsla_ann_vol > 0.4:
    vol_level = "高"
else:
    vol_level = "中等"

if sharpe > 1.0:
    sharpe_verdict = "优秀（每单位风险回报丰厚）"
elif sharpe > 0.5:
    sharpe_verdict = "一般（每单位风险回报尚可）"
else:
    sharpe_verdict = "较差（风险补偿不足）"

if tsla_cagr > sp500_cagr and sharpe > 0.5:
    overall = "高风险高回报"
elif tsla_cagr > sp500_cagr:
    overall = "高风险、回报尚可"
else:
    overall = "高风险低回报"

print("\n【判断性结论】")
print(f"  ① 收益：特斯拉过去5年 CAGR 为 {tsla_cagr*100:.1f}%（基于 {len(tsla_daily)} 个实际交易日），")
print(f"          同期 S&P 500 CAGR 为 {sp500_cagr*100:.1f}%，")
print(f"          → TSLA {outperform}大盘 {abs(cagr_gap):.1f} 个百分点。")

print(f"\n  ② 风险：TSLA 年化波动率 {tsla_ann_vol*100:.1f}%，S&P 500 为 {sp500_ann_vol*100:.1f}%，")
print(f"          TSLA 波动率约为大盘的 {vol_ratio:.1f} 倍，属于{vol_level}风险资产。")

print(f"\n  ③ 回撤：历史最大回撤为 {max_drawdown*100:.1f}%（发生于 {min_date.date()}），")
print(f"          若在高点买入，账面亏损最多曾达到这一比例，需要极强的风险承受能力。")

print(f"\n  ④ 胜率：TSLA 日胜率 {tsla_win_rate*100:.1f}%（S&P 500 为 {sp500_win_rate*100:.1f}%），")
print(f"          大约每 {round(1/tsla_win_rate):.0f} 个交易日中有 {round(tsla_win_rate*round(1/tsla_win_rate)):.0f} 天是上涨的。")

print(f"\n  ⑤ Sharpe：标准年化 Sharpe 为 {sharpe:.2f}，属于{sharpe_verdict}。")

print(f"\n  ⑥ 综合：TSLA 是一支典型的「{overall}」成长股，")
print(f"          其波动幅度远超大盘，适合风险偏好较高、能承受大幅回撤的投资者；")
print(f"          对于追求稳健回报的投资者，S&P 500 指数基金是更合适的选择。")

print()
print("=" * 60)
print("所有图表已保存至 output/ 文件夹 (7 张 PNG)")
print("All charts saved to the 'output/' folder.")
print("=" * 60)
