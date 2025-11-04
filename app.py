import os
import requests
import streamlit as st
import yfinance as yf

# ========== CONFIG ==========
st.set_page_config(page_title="Monreon Stock AI", page_icon="📈", layout="wide")

# ======== SECRETS / ENV ========
def get_secret(name: str, default: str = "") -> str:
    # try streamlit secrets first
    if name in st.secrets:
        return st.secrets[name]
    # fallback to env
    return os.getenv(name, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
GUMROAD_PRODUCT_PERMALINK = get_secret("GUMROAD_PRODUCT_PERMALINK", "gtkwix")  # your product

# ======== GUMROAD VERIFY ========
def verify_gumroad_license(license_key: str) -> bool:
    """
    Calls Gumroad's license API to check if the key is valid.
    Docs: https://gumroad.com/api#verify-license
    """
    url = "https://api.gumroad.com/v2/licenses/verify"
    payload = {
        "product_permalink": GUMROAD_PRODUCT_PERMALINK,
        "license_key": license_key,
        "increment_uses_count": True,
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        data = resp.json()
        # Gumroad returns { success: true/false, ... }
        return bool(data.get("success"))
    except Exception as e:
        st.error(f"Couldn't reach Gumroad: {e}")
        return False

# ======== OPENAI CALL ========
def ai_comment_on_stock(ticker: str, info: dict) -> str:
    """
    Simple AI layer to explain the stock in plain English.
    Uses your OpenAI key from secrets.
    """
    if not OPENAI_API_KEY:
        return "⚠️ No OpenAI API key found in secrets. Add OPENAI_API_KEY."

    # build a short context with price etc
    current_price = info.get("current_price")
    long_name = info.get("long_name") or ticker.upper()

    prompt = f"""
You are an AI stock helper. The user is considering stock {long_name} ({ticker.upper()}).
Current price: {current_price}

Give a short 3–5 bullet insight: possible trend, risk, and what to watch.
Do NOT give financial advice, just analysis.
"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You analyze stocks in a neutral, educational way."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=250,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI request failed: {e}"

# ======== STOCK FETCH ========
def fetch_stock_data(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        info = stock.info
        price = None
        if "regularMarketPrice" in info:
            price = info["regularMarketPrice"]
        elif not hist.empty:
            price = float(hist["Close"][-1])

        return {
            "ticker": ticker.upper(),
            "long_name": info.get("longName"),
            "current_price": price,
            "history": hist,
        }
    except Exception as e:
        st.error(f"Couldn't fetch {ticker}: {e}")
        return None

# ======== UI START ========
st.title("📈 Monreon Stock Market AI")
st.write("AI-powered stock insights. License-locked via Gumroad.")

# keep license in session so user doesn’t re-enter
if "license_ok" not in st.session_state:
    st.session_state.license_ok = False

if not st.session_state.license_ok:
    st.subheader("🔑 Enter your license key")
    lic = st.text_input("Paste the license key you got from Gumroad", type="password")

    if st.button("Verify license"):
        if lic.strip():
            valid = verify_gumroad_license(lic.strip())
            if valid:
                st.session_state.license_ok = True
                st.success("✅ License verified. Welcome!")
                st.rerun()
            else:
                st.error("❌ License not valid for this product. Check your Gumroad receipt.")
        else:
            st.warning("Please paste a license key.")
    st.stop()  # do not show the rest

# ======== MAIN APP (unlocked) ========
st.success("✅ License active")

st.write("Enter 1 or more tickers (comma-separated): e.g. `AAPL, TSLA, NVDA`")
tickers_input = st.text_input("Tickers", "AAPL")

if st.button("Analyze"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.warning("Add at least one ticker.")
    else:
        for t in tickers:
            st.markdown(f"### {t}")
            data = fetch_stock_data(t)
            if not data:
                st.warning(f"Skipped {t}")
                continue

            # show price
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Current price", data.get("current_price", "N/A"))
                if data.get("long_name"):
                    st.caption(data["long_name"])
            with col2:
                if data.get("history") is not None and not data["history"].empty:
                    st.line_chart(data["history"]["Close"])
                else:
                    st.write("No recent price history.")

            # AI insight
            with st.spinner("Calling AI..."):
                ai_text = ai_comment_on_stock(t, data)
            st.markdown("**AI insight:**")
            st.write(ai_text)
            st.divider()

# footer
st.caption("Monreon AI • secured with Gumroad licensing")
