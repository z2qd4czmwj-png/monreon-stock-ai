import os, datetime as dt, requests, pandas as pd, yfinance as yf, streamlit as st
from typing import Dict, Any, List

# ─────────────────────────────
# 1. CONFIG / SECRETS
# ─────────────────────────────
GUMROAD_PRODUCT_PERMALINK = st.secrets.get("GUMROAD_PRODUCT_PERMALINK", "aikbve")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
MAX_USES_PER_DAY = int(st.secrets.get("MAX_USES_PER_DAY", 50))

SESSION_AUTH, SESSION_LICENSE, SESSION_DAY, SESSION_USECOUNT = (
    "monreon_auth_ok",
    "monreon_license",
    "monreon_day",
    "monreon_usecount",
)

# ─────────────────────────────
# 2. GUMROAD LICENSE VERIFY  ✅ (the working simple version)
# ─────────────────────────────
def verify_gumroad_license(license_key: str) -> Dict[str, Any]:
    url = "https://api.gumroad.com/v2/licenses/verify"
    payload = {
        "product_permalink": GUMROAD_PRODUCT_PERMALINK,
        "license_key": license_key.strip(),
    }
    try:
        r = requests.post(url, data=payload, timeout=10)
        return r.json()
    except Exception as e:
        return {"success": False, "message": f"Request failed: {e}"}

# ─────────────────────────────
# 3. STOCK DATA HELPERS
# ─────────────────────────────
TOP_US_PRESET = ["AAPL","TSLA","NVDA","MSFT","AMZN","META","GOOGL","AMD","AVGO","JPM"]

def fetch_yf_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        return df.dropna(how="all") if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def calc_momentum(df): 
    if df.empty: return {"error":"No data"}
    c = df["Close"]; last=float(c.iloc[-1])
    week=(last-c.iloc[-5])/c.iloc[-5]*100 if len(c)>5 else None
    month=(last-c.iloc[-21])/c.iloc[-21]*100 if len(c)>21 else None
    return {"last_price":last,"1w_change_pct":week,"1m_change_pct":month}

def calc_moving_avgs(df):
    if df.empty: return {"error":"No data"}
    c=df["Close"]; res={"last_price":float(c.iloc[-1])}
    for n in [5,20,50,100,200]:
        if len(c)>=n: res[f"SMA_{n}"]=float(c.rolling(n).mean().iloc[-1])
    return res

def calc_volatility(df):
    if df.empty: return {"error":"No data"}
    r=df["Close"].pct_change().dropna()
    d=float(r.std()); a=d*(252**0.5)
    return {"daily_volatility":d,"annualized_volatility":a}

def fetch_fundamentals(t):
    try:
        i=yf.Ticker(t).fast_info
        return {"last_price":i.get("lastPrice"),
                "market_cap":i.get("marketCap"),
                "year_high":i.get("yearHigh"),
                "year_low":i.get("yearLow"),
                "currency":i.get("currency")}
    except Exception: return {"error":"No fundamentals"}

def ai_commentary(ticker,metrics,mode):
    if not OPENAI_API_KEY: return "AI commentary disabled."
    try:
        from openai import OpenAI
        client=OpenAI(api_key=OPENAI_API_KEY)
        prompt=f"Analyze {ticker} in mode {mode} using: {metrics}. Give 3 key insights."
        r=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=150,temperature=0.4)
        return r.choices[0].message.content.strip()
    except Exception as e: return f"AI failed: {e}"

# ─────────────────────────────
# 4. PAGE + LOGIN UNDER TITLE
# ─────────────────────────────
st.set_page_config(page_title="Monreon Stock AI", layout="wide")
today=dt.date.today().isoformat()
if st.session_state.get(SESSION_DAY)!=today:
    st.session_state[SESSION_DAY]=today
    st.session_state[SESSION_USECOUNT]=0
if SESSION_AUTH not in st.session_state: st.session_state[SESSION_AUTH]=False

st.title("📈 Monreon Stock AI — Advanced Market Scanner")
st.caption("AI-powered research • Gumroad license protected")

if not st.session_state[SESSION_AUTH]:
    st.subheader("🔐 Enter your Gumroad license key")
    lic=st.text_input("License key", type="password")
    if st.button("Unlock access"):
        res=verify_gumroad_license(lic)
        if res.get("success"):
            st.session_state[SESSION_AUTH]=True
            st.session_state[SESSION_LICENSE]=lic.strip()
            st.success("✅ License verified — welcome!")
            st.rerun()
        else:
            st.error(res.get("message","Invalid license."))
    st.stop()

if st.session_state.get(SESSION_USECOUNT,0)>=MAX_USES_PER_DAY:
    st.error("Daily limit reached. Come back tomorrow."); st.stop()

# ─────────────────────────────
# 5. MARKET SNAPSHOT
# ─────────────────────────────
st.markdown("### 🏦 Market Snapshot")
idx={"S&P 500":"^GSPC","NASDAQ":"^IXIC","Dow Jones":"^DJI"}
cols=st.columns(len(idx))
for i,(n,sym) in enumerate(idx.items()):
    df=fetch_yf_data(sym,"5d","1d")
    if not df.empty:
        latest,prev=df["Close"].iloc[-1],df["Close"].iloc[-2]
        pct=(latest-prev)/prev*100
        cols[i].metric(n,f"${latest:,.2f}",f"{pct:+.2f}%")
st.divider()

# ─────────────────────────────
# 6. USER INPUTS
# ─────────────────────────────
mode=st.radio("Mode",["Manual input","Top 10 US Stocks"],horizontal=True)
tickers_raw=", ".join(TOP_US_PRESET) if mode!="Manual input" else st.text_input("Tickers","AAPL, TSLA, NVDA")
period_map={"6 months (1d)":("6mo","1d"),"1 month (1d)":("1mo","1d"),"5 days (15m)":("5d","15m"),"1 day (5m)":("1d","5m")}
c1,c2=st.columns(2)
period_label=c1.selectbox("Timeframe",list(period_map.keys()))
analysis=c2.selectbox("Analysis",["Momentum","Moving Averages","Volatility","Fundamentals","AI Summary"])
period,interval=period_map[period_label]
run=st.button("🚀 Analyze now")

# ─────────────────────────────
# 7. ANALYSIS
# ─────────────────────────────
rows=[]
if run:
    st.session_state[SESSION_USECOUNT]+=1
    for t in [x.strip().upper() for x in tickers_raw.split(",") if x.strip()]:
        st.subheader(f"📊 {t}")
        df=fetch_yf_data(t,period,interval)
        if df.empty: st.warning("No data."); continue
        cA,cB=st.columns([3,1]); cA.line_chart(df["Close"]); cB.bar_chart(df["Volume"].tail(50))

        if analysis=="Momentum": m=calc_momentum(df)
        elif analysis=="Moving Averages": m=calc_moving_avgs(df)
        elif analysis=="Volatility": m=calc_volatility(df)
        elif analysis=="Fundamentals": m=fetch_fundamentals(t)
        else: m={"last_close":float(df["Close"].iloc[-1])}

        if "error" in m: st.error(m["error"])
        else: st.write(m)

        comment=ai_commentary(t,m,analysis)
        st.info(comment)
        rows.append({"ticker":t,**m,"ai_commentary":comment})
        st.markdown("---")

if rows:
    out=pd.DataFrame(rows)
    st.download_button("⬇️ Download CSV",out.to_csv(index=False).encode("utf-8"),
                       file_name="monreon_stock_ai_results.csv",mime="text/csv")

st.caption("© 2025 Monreon AI — License required. Sharing and selling keys is prohibited.")
