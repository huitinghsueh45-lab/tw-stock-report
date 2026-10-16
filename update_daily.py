"""每日更新：用固定模型權重，重算最近 10 個交易日（約兩週）的預測，並更新 real_data.json。
由 GitHub Actions 在平日盤後執行；也可在本機或 Colab 手動執行（需先放好 real_data.json）。"""
import os, sys, json, time, datetime as dt
from collections import Counter
import numpy as np, pandas as pd, requests

URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "")
FILE, KEEP, WIN = "real_data.json", 10, 20          # KEEP：保留最近幾個交易日（10 日 ≈ 兩週）
END = dt.date.today(); START = END - dt.timedelta(days=170)
S, E = START.isoformat(), END.isoformat()
FEATS = ["nc", "amp", "vr", "s5", "s20", "fo", "tr", "mg"]   # 與 Colab 訓練時完全相同

def fetch(dataset, sid):
    params = {"dataset": dataset, "data_id": sid, "start_date": S, "end_date": E}
    if TOKEN: params["token"] = TOKEN
    for k in range(3):
        j = requests.get(URL, params=params, timeout=60).json()
        if j.get("status") == 200: break
        if j.get("status") == 402: sys.exit("FinMind 請求額度已用完，本次不更新。")
        time.sleep(5)
    else:
        raise RuntimeError(f"{dataset} {sid} 失敗：{j.get('msg')}")
    time.sleep(0.3)
    return pd.DataFrame(j["data"])

def daily_table(sid):
    p = fetch("TaiwanStockPrice", sid)
    if p.empty: return None
    p["date"] = pd.to_datetime(p["date"])
    d = p.sort_values("date").set_index("date")[["Trading_Volume", "max", "min", "close"]].astype(float)
    d = d[d["Trading_Volume"] > 0]
    i = fetch("TaiwanStockInstitutionalInvestorsBuySell", sid)
    for pref, cond in [("f_", lambda s: s.str.startswith("Foreign")), ("t_", lambda s: s == "Investment_Trust")]:
        if i.empty: d[pref+"buy"] = 0.0; d[pref+"sell"] = 0.0; continue
        x = i[cond(i["name"])].copy(); x["date"] = pd.to_datetime(x["date"])
        g = x.groupby("date")[["buy", "sell"]].sum()
        d[pref+"buy"] = g["buy"].reindex(d.index).fillna(0.0); d[pref+"sell"] = g["sell"].reindex(d.index).fillna(0.0)
    m = fetch("TaiwanStockMarginPurchaseShortSale", sid)
    if m.empty: d["mchg"] = 0.0
    else:
        m["date"] = pd.to_datetime(m["date"]); m = m.set_index("date")
        d["mchg"] = (m["MarginPurchaseTodayBalance"] - m["MarginPurchaseYesterdayBalance"]).reindex(d.index).fillna(0.0)
    vol = d["Trading_Volume"]
    d["amp"] = (d["max"] - d["min"]) / d["close"]
    d["vr"] = vol / vol.rolling(20).mean()
    d["s5"] = d["close"].rolling(5).mean().pct_change(); d["s20"] = d["close"].rolling(20).mean().pct_change()
    d["fo"] = (d["f_buy"] - d["f_sell"]) / vol; d["tr"] = (d["t_buy"] - d["t_sell"]) / vol
    d["mg"] = d["mchg"] * 1000 / vol
    return d

def window_x(d, t):
    w = d.iloc[t - WIN + 1: t + 1]
    nc = w["close"].values / w["close"].values[0] - 1
    return np.column_stack([nc] + [w[c].values for c in FEATS[1:]])

J = json.load(open(FILE, encoding="utf-8")); M = J["model"]
mean, std, coef, bias = (np.array(M[k]) for k in ("mean", "std", "coef", "intercept"))
def probs(x):
    lg = coef @ ((x.reshape(-1) - mean) / std) + bias; e = np.exp(lg - lg.max()); return e / e.sum()

tabs = {}
for s in J["stocks"]:
    try:
        d = daily_table(s["id"])
        if d is not None and len(d) >= 75 and np.isfinite(window_x(d, len(d) - 1)).all(): tabs[s["id"]] = d
        else: print("略過（資料不足）", s["id"])
    except Exception as e: print("略過", s["id"], str(e)[:80])
if not tabs: sys.exit("沒有取得任何資料，不更新。")
ref = Counter(tuple(d.index[-70:].strftime("%Y-%m-%d")) for d in tabs.values()).most_common(1)[0][0]
asof = ref[-1]
if "rdays" in J and asof <= J["asof"]:
    print(f"沒有新的交易日資料（最新 {asof}），不更新。"); sys.exit(0)

stocks = []
for s in J["stocks"]:
    d = tabs.get(s["id"])
    if d is None or tuple(d.index[-70:].strftime("%Y-%m-%d")) != ref: print("日期不齊，略過", s["id"]); continue
    P = []
    for o in range(KEEP - 1, -1, -1):
        p = probs(window_x(d, len(d) - 1 - o)); u, dn = round(float(p[0]) * 100, 1), round(float(p[2]) * 100, 1)
        P.append([u, round(100 - u - dn, 1), dn])
    lots = lambda b, sl: [[int(round(x / 1000)), int(round(y / 1000)), int(round((x - y) / 1000))] for x, y in zip(d[b].iloc[-(KEEP + 4):], d[sl].iloc[-(KEEP + 4):])]
    stocks.append({"id": s["id"], "name": s["name"], "probs": P[-1], "p": P,
        "x": [round(float(v), 6) for v in window_x(d, len(d) - 1).reshape(-1)],
        "px": [round(float(v), 2) for v in d["close"].iloc[-70:]],
        "vo": [round(float(v), 3) for v in d["vr"].iloc[-(KEEP + 19):]],
        "fo": [round(float(v) * 100, 3) for v in d["fo"].iloc[-(KEEP + 19):]],
        "tr": [round(float(v) * 100, 3) for v in d["tr"].iloc[-(KEEP + 19):]],
        "fl": lots("f_buy", "f_sell"), "tl": lots("t_buy", "t_sell")})
J.update(asof=asof, dates=list(ref), rdays=list(ref[-KEEP:]), stocks=stocks,
         updated=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M（台灣時間）"))
json.dump(J, open(FILE, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("已更新", asof, len(stocks), "檔，保留", KEEP, "個交易日")
