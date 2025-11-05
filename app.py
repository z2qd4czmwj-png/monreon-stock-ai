import streamlit as st
import requests
import pandas as pd
import yfinance as yf

# ====== YOUR SETTINGS ======
# you already know this one:
GUMROAD_PRODUCT_PERMALINK = "aikbve"   # what you have now
# if one day you find the product_id, put it here:
GUMROAD_PRODUCT_ID = ""                # leave empty for now
MAX_USES_PER_DAY = 50
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

st.set_page_config(page_title="Monreon Stock AI", layout="wide")

# ====== 1. LOGIN BOX ======
st.title("📈 Monreon Stock AI")
st.caption("Gumroad-locked tool")

# session init
if "auth" not in st.session_state:
    st.session_state["auth"] = False
if "uses" not in st.session_state:
    st.session_state["uses"] = 0

def verify_gumroad(lic: str):
    url = "https://api.gumroad.com/v2/licenses/verify"
    payload = {"license_key": lic.strip()}
    # try product_id first if we ever add it
    if GUMROAD_PRODUCT_ID:
        payload["product_id"] = GUMROAD_PRODUCT_ID
    else:
        payload["product_permalink"] = GUMROAD_PRODUCT_PERMALINK
    r = requests.post(url, data=payload, timeout=10)
    return r.json(), payload

if not st.session_state["auth"]:
    st.subheader("🔐 Enter your Gumroad license")
    lic = st.text_input("License key", type="password")

    if st.button("Unlock"):
        data, sent_payload = verify_gumroad(lic)

        # show what we sent + what we got (so we see the real problem)
        st.markdown("**Request we sent to Gumroad:**")
        st.code(sent_payload, language="json")
        st.markdown("**Response we got from Gumroad:**")
        st.code(data, language="json")

        if data.get("success"):
            st.session_state["auth"] = True
            st.success("✅ License verified!")
            st.experimental_rerun()
        else:
            st.error(data.get("message", "License not valid."))
    st.stop()

# ====== 2. DAILY LIMIT ======
if st.session_state["uses"] >= MAX_USES_PER_DAY:
    st.error("Daily limit reached for this key.")
    st.stop()

# ====== 3. THE TOOL (same as before but simpler) ======
st.markdown("### 🏦 Market snapshot")
idx = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI"}
cols = st.columns(3)
for i, (name, sym) in enumerate(idx.items()):
    df = yf.download(sym, period="5d", interval="1d", progress=False)
    if not df.empty:
        latest = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2]
        pct = (latest - prev) / prev * 100
        cols[i].metric(name, f"${latest:,.2f}", f"{pct:+.2f}%")
    else:
        cols[i].write(name)

st.divider()

input_mode = st.radio("Input mode", ["Manual", "Top 10 US"], horizontal=True)
if input_mode == "Manual":
    tickers_raw = st.text_input("Tickers", "AAPL, TSLA, NVDA")
else:
    tickers_raw = "AAPL, TSLA, NVDA, MSFT, AMZN, META, GOOGL, AMD, AVGO, JPM"
    st.info("Top 10 prefilled.")

period = st.selectbox("Period", ["6mo", "3mo", "1mo", "5d"], index=0)

if st.button("🚀 Analyze"):
    st.session_state["uses"] += 1
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    rows = []
    for t in tickers:
        st.subheader(f"📊 {t}")
        df = yf.download(t, period=period, interval="1d", progress=False)
        if df.empty:
            st.warning("No data.")
            continue
        c1, c2 = st.columns([3,1])
        c1.line_chart(df["Close"])
        c2.bar_chart(df["Volume"].tail(50))

        rows.append({"ticker": t, "last_price": float(df["Close"].iloc[-1])})

    if rows:
        out = pd.DataFrame(rows)
        st.download_button(
            "⬇️ Download CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="monreon_stock_ai.csv",
            mime="text/csv",
        )

st.caption("© 2025 Monreon AI")
