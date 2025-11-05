import os
import datetime
import requests
import streamlit as st
import yfinance as yf
import pandas as pd

# =========================
# SECRETS / CONFIG
# =========================
def get_secret(section: str, key: str, default: str = "") -> str:
    try:
        if section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except Exception:
        pass
    return os.getenv(key, default)

GUMROAD_PRODUCT_PERMALINK = get_secret("gumroad", "PRODUCT_PERMALINK", "")
GUMROAD_ACCESS_TOKEN      = get_secret("gumroad", "ACCESS_TOKEN", "")
OPENAI_API_KEY            = get_secret("openai", "OPENAI_API_KEY", "")
MAX_USES_PER_DAY          = int(get_secret("app", "MAX_USES_PER_DAY", "50"))

# =========================
# GUMROAD LICENSE CHECK
# =========================
def verify_gumroad_license(license_key: str):
    if not GUMROAD_PRODUCT_PERMALINK:
        return False, {"message": "Missing Gumroad product permalink."}

    url = "https://api.gumroad.com/v2/licenses/verify"
    payload = {
        "product_permalink": GUMROAD_PRODUCT_PERMALINK,
        "license_key": license_key,
    }
    if GUMROAD_ACCESS_TOKEN:
        payload["access_token"] = GUMROAD_ACCESS_TOKEN

    try:
        resp = requests.post(url, data=payload, timeout=10)
        data = resp.json()
        return data.get("success", False), data
    except Exception as e:
        return False, {"message": f"API error: {e}"}


# =========================
# SESSION / DAILY USAGE
# =========================
def init_state():
    if "license_valid" not in st.session_state:
        st.session_state.license_valid = False
        st.session_state.license_data = {}
    if "uses_today" not in st.session_state:
        st.session_state.uses_today = 0
    if "uses_date" not in st.session_state:
        st.session_state.uses_date = datetime.date.today().isoformat()


def reset_if_new_day():
    today = datetime.date.today().isoformat()
    if st.session_state.get("uses_date") != today:
        st.session_state.uses_today = 0
        st.session_state.uses_date = today


def bump_usage():
    st.session_state.uses_today += 1


# =========================
# DATA + ANALYSIS HELPERS
# =========================
def fetch_price_history(ticker: str, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


def calc_momentum(df):
    if df.empty:
        return {"error": "No data"}
    close = df["Close"]
    latest = close.iloc[-1]
    week = (latest - close.iloc[-5]) / close.iloc[-5] * 100 if len(close) > 5 else None
    month = (latest - close.iloc[-21]) / close.iloc[-21] * 100 if len(close) > 21 else None
    return {"last_price": latest, "1w_change_pct": week, "1m_change_pct": month}


def calc_moving_avgs(df):
    if df.empty:
        return {"error": "No data"}
    close = df["Close"]
    out = {"last_price": close.iloc[-1]}
    for win in [5, 20, 50, 100, 200]:
        if len(close) >= win:
            out[f"SMA_{win}"] = close.rolling(win).mean().iloc[-1]
    return out


def calc_volatility(df):
    if df.empty:
        return {"error": "No data"}
    returns = df["Close"].pct_change().dropna()
    daily = returns.std()
    ann = daily * (252 ** 0.5)
    return {"daily_volatility": float(daily), "annualized_volatility": float(ann)}


def fetch_fundamentals(ticker: str):
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


def ai_commentary(ticker, metrics, mode):
    if not OPENAI_API_KEY:
        return "AI commentary unavailable (no OpenAI key configured)."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            f"You are a professional market analyst. "
            f"Summarize {ticker}'s outlook based on: {metrics}. "
            f"Use a confident tone and highlight sentiment (bullish/bearish/neutral). "
            f"Give 3 concise actionable points."
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
# MARKET TOP 10
# =========================
def top_10_traded():
    """
    Returns a static list (YFinance doesn’t provide live volume screeners)
    """
    return ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOG", "NFLX", "AMD", "INTC"]


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Monreon Stock AI — Advanced Market Scanner", layout="wide")
init_state()
reset_if_new_day()

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("### 🔐 Monreon AI Login")
    license_key = st.text_input("License key", type="password")
    if st.button("Unlock"):
        ok, data = verify_gumroad_license(license_key)
        if ok:
            st.session_state.license_valid = True
            st.session_state.license_data = data
            st.success("✅ License verified!")
        else:
            st.error(data.get("message", "License invalid."))

    st.write(f"Today: {st.session_state.uses_today} / {MAX_USES_PER_DAY}")

    if not st.session_state.license_valid:
        st.stop()
    if st.session_state.uses_today >= MAX_USES_PER_DAY:
        st.error("Daily usage limit reached.")
        st.stop()

# ---- HEADER ----
st.title("📊 Monreon Stock AI — Advanced Market Scanner")
st.caption("Smarter analysis powered by AI + market data")

# Small market overview
st.markdown("#### 🏦 Market Snapshot")
major_indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI"}
cols = st.columns(3)
for i, (name, symbol) in enumerate(major_indices.items()):
    df = fetch_price_history(symbol, period="5d")
    if not df.empty:
        latest = df["Close"].iloc[-1]
        change = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
        cols[i].metric(name, f"${latest:,.2f}", f"{change:+.2f}%")

st.divider()

# ---- USER INPUT ----
preset = st.radio("Choose input mode:", ["Manual input", "Top 10 Most Traded US Stocks"])
if preset == "Manual input":
    tickers_raw = st.text_input("Enter tickers (comma separated)", "AAPL, TSLA, NVDA")
else:
    tickers_raw = ", ".join(top_10_traded())
    st.info("Auto-filled with Top 10 Most Traded Stocks")

periods = {
    "6 months (1d)": ("6mo", "1d"),
    "1 month (1d)": ("1mo", "1d"),
    "5 days (15m)": ("5d", "15m"),
    "1 day (5m)": ("1d", "5m"),
}
col1, col2 = st.columns([2, 2])
with col1:
    period_label = st.selectbox("Timeframe", list(periods.keys()))
with col2:
    mode = st.selectbox(
        "Analysis mode",
        [
            "Find momentum",
            "Check moving averages",
            "Check volatility",
            "Quick fundamentals",
            "AI summary (if OpenAI key)",
        ],
    )

period, interval = periods[period_label]
if st.button("🔍 Analyze Now", type="primary"):
    bump_usage()
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    all_rows = []

    for ticker in tickers:
        st.subheader(f"📈 {ticker}")
        df = fetch_price_history(ticker, period=period, interval=interval)

        if df.empty:
            st.warning("No data available.")
            continue

        # Chart with price + volume
        chart_cols = st.columns([3, 1])
        with chart_cols[0]:
            st.line_chart(df[["Close"]], height=220)
        with chart_cols[1]:
            st.bar_chart(df[["Volume"]], height=220)

        if mode == "Find momentum":
            metrics = calc_momentum(df)
        elif mode == "Check moving averages":
            metrics = calc_moving_avgs(df)
        elif mode == "Check volatility":
            metrics = calc_volatility(df)
        elif mode == "Quick fundamentals":
            metrics = fetch_fundamentals(ticker)
        else:
            metrics = {"last_close": float(df["Close"].iloc[-1])}

        if "error" in metrics:
            st.error(metrics["error"])
            continue

        st.write(metrics)
        summary = ai_commentary(ticker, metrics, mode)
        st.info(summary)

        row = {"ticker": ticker, "mode": mode, **metrics, "ai_commentary": summary}
        all_rows.append(row)
        st.divider()

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        st.download_button(
            "⬇️ Download results as CSV",
            data=out_df.to_csv(index=False).encode(),
            file_name="monreon_stock_ai_results.csv",
            mime="text/csv",
        )

st.caption("© 2025 Monreon AI. Licensed for personal/client use only. Redistribution or sharing of license keys is prohibited.")
