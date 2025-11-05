# app.py — Monreon Stock AI (Gumroad under-title login + advanced tool)
import os
import datetime as dt
import time
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# =========================
# 1. SECRETS / CONFIG
# =========================
def get_secret(section: str, key: str, default: str = "") -> str:
    try:
        if section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except Exception:
        pass
    return os.getenv(key, default)

GUMROAD_PRODUCT_PERMALINK = get_secret("gumroad", "PRODUCT_PERMALINK", "").strip()
GUMROAD_ACCESS_TOKEN      = get_secret("gumroad", "ACCESS_TOKEN", "").strip()
OPENAI_API_KEY            = get_secret("openai", "OPENAI_API_KEY", "").strip()

# read daily limit from either [app] or root
raw_limit = (
    get_secret("app", "MAX_USES_PER_DAY", "")
    or str(st.secrets.get("MAX_USES_PER_DAY", "50"))
)
try:
    MAX_USES_PER_DAY = int(raw_limit)
except Exception:
    MAX_USES_PER_DAY = 50

SESSION_AUTH     = "monreon_auth_ok"
SESSION_LICENSE  = "monreon_license"
SESSION_DAY      = "monreon_day"
SESSION_USECOUNT = "monreon_usecount"
SESSION_LAST_ERR = "monreon_last_error"

# =========================
# 2. GUMROAD VERIFY (same style as when it worked)
# =========================
def verify_gumroad_license(license_key: str) -> Dict[str, Any]:
    """
    This is the simple version: send product_permalink + license_key
    + access_token (if you have it).
    This is the one you used when it worked.
    """
    url = "https://api.gumroad.com/v2/licenses/verify"
    payload = {
        "product_permalink": GUMROAD_PRODUCT_PERMALINK,
        "license_key": license_key.strip(),
    }
    if GUMROAD_ACCESS_TOKEN:
        payload["access_token"] = GUMROAD_ACCESS_TOKEN

    try:
        r = requests.post(url, data=payload, timeout=10)
        data = r.json()
        return data
    except Exception as e:
        return {"success": False, "message": f"Request to Gumroad failed: {e}"}

# =========================
# 3. DATA HELPERS
# =========================
TOP_US_PRESET = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "AMD", "AVGO", "JPM"]

def fetch_yf_data(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.dropna(how="all", inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def calc_momentum(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"error": "No price data"}
    close = df["Close"]
    latest = float(close.iloc[-1])
    week = None
    month = None
    if len(close) > 5:
        week = (latest - close.iloc[-5]) / close.iloc[-5] * 100
    if len(close) > 21:
        month = (latest - close.iloc[-21]) / close.iloc[-21] * 100
    return {
        "last_price": latest,
        "1w_change_pct": week,
        "1m_change_pct": month,
    }

def calc_moving_avgs(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"error": "No price data"}
    close = df["Close"]
    out = {"last_price": float(close.iloc[-1])}
    for win in [5, 20, 50, 100, 200]:
        if len(close) >= win:
            out[f"SMA_{win}"] = float(close.rolling(win).mean().iloc[-1])
    return out

def calc_volatility(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"error": "No price data"}
    ret = df["Close"].pct_change().dropna()
    daily = float(ret.std())
    annual = daily * (252 ** 0.5)
    return {"daily_volatility": daily, "annualized_volatility": annual}

def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    try:
        info = yf.Ticker(ticker).fast_info
        return {
            "last_price": info.get("lastPrice"),
            "market_cap": info.get("marketCap"),
            "year_high": info.get("yearHigh"),
            "year_low": info.get("yearLow"),
            "currency": info.get("currency"),
        }
    except Exception:
        return {"error": "Fundamentals unavailable"}

def ai_commentary(ticker: str, metrics: Dict[str, Any], mode: str) -> str:
    if not OPENAI_API_KEY:
        return "AI commentary disabled (no OpenAI key set)."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            f"Analyze {ticker} for a trader. Mode: {mode}. "
            f"Metrics: {metrics}. "
            f"Give 3 actionable bullets. Mark overall tone as bullish/bearish/neutral."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI failed: {e}"

# =========================
# 4. INIT SESSION
# =========================
st.set_page_config(page_title="Monreon Stock AI", layout="wide")

today = dt.date.today().isoformat()
if SESSION_DAY not in st.session_state:
    st.session_state[SESSION_DAY] = today
if st.session_state[SESSION_DAY] != today:
    st.session_state[SESSION_DAY] = today
    st.session_state[SESSION_USECOUNT] = 0
if SESSION_USECOUNT not in st.session_state:
    st.session_state[SESSION_USECOUNT] = 0
if SESSION_AUTH not in st.session_state:
    st.session_state[SESSION_AUTH] = False

# =========================
# 5. PAGE HEADER + LOGIN (in main, not sidebar)
# =========================
st.title("📈 Monreon Stock AI — Advanced Market Scanner")
st.caption("AI-style stock research with Gumroad-protected access.")

# show login box if not authed
if not st.session_state[SESSION_AUTH]:
    st.subheader("🔐 Unlock your tool")
    if not GUMROAD_PRODUCT_PERMALINK:
        st.error("Gumroad product permalink is missing. In Streamlit secrets add:\n[gumroad]\nPRODUCT_PERMALINK = \"aikbve\"")
    license_input = st.text_input("Enter the license key you got from Gumroad", type="password")
    if st.button("Unlock my access"):
        data = verify_gumroad_license(license_input)
        st.session_state[SESSION_LAST_ERR] = data
        if data.get("success"):
            st.session_state[SESSION_AUTH] = True
            st.session_state[SESSION_LICENSE] = license_input.strip()
            st.success("✅ License verified. Welcome!")
            st.rerun()
        else:
            st.error(data.get("message", "License not valid for this product."))
            st.stop()

    # show raw response for you (developer) so you can see exact gumroad message
    if st.session_state.get(SESSION_LAST_ERR):
        st.code(st.session_state[SESSION_LAST_ERR], language="json")

    st.stop()

# =========================
# 6. ENFORCE DAILY LIMIT
# =========================
if st.session_state[SESSION_USECOUNT] >= MAX_USES_PER_DAY:
    st.error("You reached today's usage limit for this license. Come back tomorrow.")
    st.stop()

# =========================
# 7. MARKET SNAPSHOT
# =========================
st.markdown("### 🏦 Market Snapshot")
indices = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
}
cols = st.columns(len(indices))
for i, (name, symbol) in enumerate(indices.items()):
    df_i = fetch_yf_data(symbol, period="5d", interval="1d")
    if not df_i.empty:
        latest = df_i["Close"].iloc[-1]
        prev = df_i["Close"].iloc[-2] if len(df_i) > 1 else latest
        pct = (latest - prev) / prev * 100 if prev else 0
        cols[i].metric(name, f"${latest:,.2f}", f"{pct:+.2f}%")
    else:
        cols[i].write(name)

st.divider()

# =========================
# 8. USER INPUT AREA
# =========================
preset_mode = st.radio("Choose input mode:", ["Manual input", "Top 10 Most Traded US Stocks"], horizontal=True)

if preset_mode == "Manual input":
    tickers_raw = st.text_input("Enter tickers", "AAPL, TSLA, NVDA")
else:
    tickers_raw = ", ".join(TOP_US_PRESET)
    st.info("Filled with top US tickers.")

period_choices = {
    "6 months (1d)": ("6mo", "1d"),
    "1 month (1d)": ("1mo", "1d"),
    "5 days (15m)": ("5d", "15m"),
    "1 day (5m)": ("1d", "5m"),
}
col1, col2 = st.columns(2)
with col1:
    period_label = st.selectbox("Timeframe", list(period_choices.keys()), index=0)
with col2:
    analysis_mode = st.selectbox(
        "Analysis mode",
        [
            "Find momentum",
            "Check moving averages",
            "Check volatility",
            "Quick fundamentals",
            "AI summary (if OpenAI key)",
        ],
    )

period, interval = period_choices[period_label]

run_btn = st.button("🚀 Analyze now", type="primary")

# =========================
# 9. RUN ANALYSIS
# =========================
all_rows: List[Dict[str, Any]] = []

if run_btn:
    # count usage
    st.session_state[SESSION_USECOUNT] += 1

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    for ticker in tickers:
        st.subheader(f"📊 {ticker}")
        df = fetch_yf_data(ticker, period=period, interval=interval)

        # price + volume chart
        if not df.empty:
            chart_cols = st.columns([3, 1])
            with chart_cols[0]:
                st.line_chart(df[["Close"]])
            with chart_cols[1]:
                st.bar_chart(df[["Volume"]].tail(50))
        else:
            st.warning("No data for this ticker.")
            continue

        # analysis
        if analysis_mode == "Find momentum":
            metrics = calc_momentum(df)
        elif analysis_mode == "Check moving averages":
            metrics = calc_moving_avgs(df)
        elif analysis_mode == "Check volatility":
            metrics = calc_volatility(df)
        elif analysis_mode == "Quick fundamentals":
            metrics = fetch_fundamentals(ticker)
        else:  # AI summary
            metrics = {"last_close": float(df["Close"].iloc[-1])}

        if "error" in metrics:
            st.error(metrics["error"])
        else:
            st.write(metrics)

        # AI commentary
        ai_text = ai_commentary(ticker, metrics, analysis_mode)
        st.info(ai_text)

        row = {"ticker": ticker, "mode": analysis_mode, **metrics, "ai_commentary": ai_text}
        all_rows.append(row)

        st.markdown("---")

# =========================
# 10. CSV EXPORT
# =========================
if all_rows:
    out_df = pd.DataFrame(all_rows)
    st.download_button(
        "⬇️ Download results as CSV",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="monreon_stock_ai_results.csv",
        mime="text/csv",
    )

st.caption("© 2025 Monreon AI. Licensed for personal/client use only. Redistribution or sharing of license keys is prohibited.")
