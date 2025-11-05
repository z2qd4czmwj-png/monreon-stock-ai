import os
import time
import datetime as dt
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# =============== secrets helper =================
def get_secret(section: str, key: str, default: str = "") -> str:
    try:
        if section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except Exception:
        pass
    return os.getenv(key, default)

# =============== load config ====================
GUMROAD_PRODUCT_PERMALINK = get_secret("gumroad", "PRODUCT_PERMALINK", "").strip()
GUMROAD_ACCESS_TOKEN     = get_secret("gumroad", "ACCESS_TOKEN", "").strip()
MAX_USES_PER_DAY         = int(get_secret("app", "MAX_USES_PER_DAY", "50"))
SESSION_AUTH_OK          = "monreon_auth_ok"
SESSION_TODAY_COUNT      = "monreon_today_count"
SESSION_TODAY_DATE       = "monreon_today_date"
SESSION_LICENSE          = "monreon_license"
SESSION_LAST_ERROR       = "monreon_last_error"

# =============== gumroad verify =================
def verify_gumroad_license(license_key: str) -> Dict[str, Any]:
    url = "https://api.gumroad.com/v2/licenses/verify"
    payload = {
        "product_permalink": GUMROAD_PRODUCT_PERMALINK,
        "license_key": license_key.strip(),
    }
    if GUMROAD_ACCESS_TOKEN:
        payload["access_token"] = GUMROAD_ACCESS_TOKEN
    try:
        r = requests.post(url, data=payload, timeout=10)
        return r.json()
    except Exception as e:
        return {"success": False, "message": f"request failed: {e}"}

# =============== license gate ===================
def license_gate():
    today = dt.date.today().isoformat()
    if SESSION_TODAY_DATE not in st.session_state:
        st.session_state[SESSION_TODAY_DATE] = today
    if st.session_state[SESSION_TODAY_DATE] != today:
        st.session_state[SESSION_TODAY_DATE] = today
        st.session_state[SESSION_TODAY_COUNT] = 0
    if SESSION_TODAY_COUNT not in st.session_state:
        st.session_state[SESSION_TODAY_COUNT] = 0

    with st.sidebar:
        st.markdown("🔐 **Monreon AI Login**")
        if not GUMROAD_PRODUCT_PERMALINK:
            st.error("You didn't set PRODUCT_PERMALINK in secrets.\nShould be like: `aikbve`")
        license_key = st.text_input("License key", type="password")
        st.write(f"Today: {st.session_state[SESSION_TODAY_COUNT]} / {MAX_USES_PER_DAY}")

        if st.button("Unlock"):
            if not license_key:
                st.warning("Enter your license key.")
            else:
                data = verify_gumroad_license(license_key)
                # save error so we can show it
                st.session_state[SESSION_LAST_ERROR] = data
                if data.get("success"):
                    st.session_state[SESSION_AUTH_OK] = True
                    st.session_state[SESSION_LICENSE] = license_key
                    st.success("License verified ✅")
                    st.rerun()
                else:
                    st.error(data.get("message", "License not valid"))

        # show last error for debugging
        if st.session_state.get(SESSION_LAST_ERROR):
            st.code(st.session_state[SESSION_LAST_ERROR], language="json")

    if not st.session_state.get(SESSION_AUTH_OK, False):
        st.stop()

    if st.session_state[SESSION_TODAY_COUNT] >= MAX_USES_PER_DAY:
        st.error("Daily limit reached for this key.")
        st.stop()

# =============== stock helpers ==================
TOP_US_PRESET = ["AAPL","TSLA","NVDA","MSFT","AMZN","META","GOOGL","AMD","AVGO","JPM"]

def fetch_yf_data(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.dropna(how="all", inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def analyze_ticker(ticker: str):
    ticker = ticker.upper().strip()
    info = {"ticker": ticker, "ok": False}
    df = fetch_yf_data(ticker, "6mo", "1d")
    if df.empty:
        info["reason"] = "no data from yfinance"
        return info
    close = df["Close"].iloc[-1]
    info["last_price"] = float(close)
    last20 = df["Close"].tail(20)
    info["ma20"] = float(last20.mean())
    info["momentum"] = "bullish" if close > last20.mean() else "neutral/weak"
    info["last_volume"] = int(df["Volume"].iloc[-1])
    info["avg_volume_20"] = int(df["Volume"].tail(20).mean())
    info["ok"] = True
    return info

def build_dataframe(results):
    rows = []
    for r in results:
        if not r.get("ok"):
            rows.append({"ticker": r["ticker"], "status": r.get("reason","no data")})
        else:
            rows.append({
                "ticker": r["ticker"],
                "price": r["last_price"],
                "ma20": r["ma20"],
                "momentum": r["momentum"],
                "last_volume": r["last_volume"],
                "avg_volume_20": r["avg_volume_20"],
            })
    return pd.DataFrame(rows)

# =============== main app =======================
def main():
    st.set_page_config(page_title="Monreon Stock AI", layout="wide")
    license_gate()

    st.title("📈 Monreon Stock AI — Market Scanner")
    left, right = st.columns([2,1])

    with right:
        st.markdown("### Presets")
        if st.button("Top 10 US"):
            st.session_state["tickers_input"] = ", ".join(TOP_US_PRESET)

        if "last_df" in st.session_state:
            csv = st.session_state["last_df"].to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "monreon_scan.csv", "text/csv")

        st.write(f"Today used: {st.session_state[SESSION_TODAY_COUNT]} / {MAX_USES_PER_DAY}")

    with left:
        tickers_default = st.session_state.get("tickers_input", "AAPL, TSLA, NVDA")
        tickers_str = st.text_input("Tickers", tickers_default)
        st.session_state["tickers_input"] = tickers_str

        period = st.selectbox("History", ["1mo","3mo","6mo","1y"], index=2)
        interval = st.selectbox("Interval", ["1d","1h","30m","15m"], index=0)
        chart_ticker = st.text_input("Chart ticker", "AAPL")

        if st.button("Analyze", type="primary"):
            st.session_state[SESSION_TODAY_COUNT] += 1
            tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
            results = []
            with st.spinner("Scanning..."):
                for t in tickers:
                    results.append(analyze_ticker(t))
                    time.sleep(0.1)
            df = build_dataframe(results)
            st.session_state["last_df"] = df
            st.subheader("Scan result")
            st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.subheader("Chart")
        df_chart = fetch_yf_data(chart_ticker, period=period, interval=interval)
        if not df_chart.empty:
            st.line_chart(df_chart["Close"], height=250)
            st.bar_chart(df_chart[["Volume"]].tail(30), height=180)
        else:
            st.write("No data")

if __name__ == "__main__":
    main()
