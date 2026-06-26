import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Options Pricer", layout="wide", page_icon="📈")

# ── Black-Scholes ─────────────────────────────────────
def bs(S, K, T, r, sig, kind):
    if T <= 0:
        return max(0, S - K) if kind == "call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def greeks(S, K, T, r, sig):
    if T <= 0:
        return None
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)
    g = pdf_d1 / (S * sig * sqrt_T)
    return {
        "Δ Call":         round(norm.cdf(d1), 4),
        "Δ Put":          round(norm.cdf(d1) - 1, 4),
        "Γ (both)":       round(g, 6),
        "Vega /1% vol":   round(S * pdf_d1 * sqrt_T / 100, 4),
        "Θ Call /day":    round((-S * pdf_d1 * sig / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365, 4),
        "Θ Put /day":     round((-S * pdf_d1 * sig / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365, 4),
        "Rho Call /1% r": round(K * T * np.exp(-r * T) * norm.cdf(d2) / 100, 4),
        "Rho Put /1% r":  round(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100, 4),
    }

@st.cache_data(ttl=60)
def fetch_stock(ticker):
    try:
        t = yf.Ticker(ticker.upper())
        hist = t.history(period="6mo")
        if hist.empty:
            return None, None, None, None
        price = round(float(hist["Close"].iloc[-1]), 2)
        prev  = round(float(hist["Close"].iloc[-2]), 2)
        rets  = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        hvol  = round(float(rets.std() * np.sqrt(252) * 100), 1)
        name  = ticker.upper()
        try:
            fi = t.fast_info
            name = getattr(fi, "company_name", ticker.upper()) or ticker.upper()
        except Exception:
            pass
        return price, prev, hist, hvol, name
    except Exception:
        return None, None, None, None, None

def strike_step(price):
    if price < 5:     return 0.05
    if price < 20:    return 0.5
    if price < 50:    return 1.0
    if price < 200:   return 2.5
    if price < 500:   return 5.0
    if price < 1000:  return 10.0
    return 25.0

def gen_strikes(S, n=12):
    step = strike_step(S)
    base = round(round(S / step) * step, 4)
    return sorted(set(round(base + step * i, 4) for i in range(-5, n - 5) if base + step * i > 0))

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Options Pricer")
    st.caption("Live Black-Scholes for any stock")

    ticker_input = st.text_input("Stock ticker", value="SPCX",
                                 help="Try AAPL · TSLA · NVDA · MSFT · AMZN").strip().upper()

    result = fetch_stock(ticker_input)
    if result[0] is None:
        st.error(f"Could not fetch '{ticker_input}' — check the ticker.")
        st.stop()

    live_price, prev_price, hist, hvol, company_name = result
    daily_chg = round(live_price - prev_price, 2)
    daily_pct  = round(daily_chg / prev_price * 100, 2) if prev_price else 0

    if daily_chg >= 0:
        st.success(f"**${live_price:,.2f}**  ▲ +${daily_chg} (+{daily_pct}%) today")
    else:
        st.error(f"**${live_price:,.2f}**  ▼ ${daily_chg} ({daily_pct}%) today")

    st.caption(f"30-day realised vol: **{hvol}%**  ·  Updates every 60s")
    st.divider()

    st.subheader("Black-Scholes inputs")

    step = strike_step(live_price)
    lo   = max(step, round(live_price * 0.25, 4))
    hi_s = round(live_price * 4.0, 4)
    hi_k = round(live_price * 3.0, 4)

    S = st.slider("S — Stock price ($)", min_value=lo, max_value=hi_s,
                  value=float(live_price), step=step,
                  help="Current stock price. Defaults to live price — adjust to model scenarios.")

    K = st.slider("K — Strike price ($)", min_value=lo, max_value=hi_k,
                  value=float(round(round(live_price / step) * step, 4)), step=step,
                  help="The fixed price in the option contract.")

    T_days = st.slider("T — Days to expiry", min_value=1, max_value=730, value=180, step=1,
                       help="Number of calendar days until the option expires.")
    T = T_days / 365

    r_pct = st.slider("r — Risk-free rate (%)", min_value=0.0, max_value=15.0,
                      value=4.0, step=0.25,
                      help="US Treasury yield used as the risk-free rate.")
    r = r_pct / 100

    default_vol = max(10, min(int(hvol), 200)) if hvol else 30
    sig_pct = st.slider("σ — Implied volatility (%)", min_value=5, max_value=300,
                        value=default_vol, step=1,
                        help="Annualised volatility assumption. Defaults to 30-day historical vol.")
    sigma = sig_pct / 100

    st.divider()
    st.caption("All prices update instantly as you move any slider.")

# ── Main ──────────────────────────────────────────────
st.header(f"{company_name}  ·  {ticker_input}")

# Summary cards
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("S  (stock price)",  f"${S:,.2f}")
c2.metric("K  (strike)",       f"${K:,.2f}")
c3.metric("T  (days)",         T_days)
c4.metric("r  (rate)",         f"{r_pct:.2f}%")
c5.metric("σ  (vol)",          f"{sig_pct}%")
c6.metric("T  (years)",        f"{T:.4f}")

st.divider()

# Call & put for selected K
call_px  = bs(S, K, T, r, sigma, "call")
put_px   = bs(S, K, T, r, sigma, "put")
c_int    = max(0.0, S - K)
p_int    = max(0.0, K - S)

left, right = st.columns(2)
left.metric("Call price at K",  f"${call_px:.4f}",
            delta=f"Intrinsic ${c_int:.2f}  +  Time ${call_px-c_int:.4f}")
right.metric("Put price at K",  f"${put_px:.4f}",
             delta=f"Intrinsic ${p_int:.2f}  +  Time ${put_px-p_int:.4f}")

# Greeks
g = greeks(S, K, T, r, sigma)
if g:
    with st.expander("Greeks for selected K", expanded=True):
        gcols = st.columns(4)
        for i, (label, val) in enumerate(g.items()):
            gcols[i % 4].metric(label, val)

st.divider()

# Options table
st.subheader("Options table")
strikes = gen_strikes(S)
atm_k   = min(strikes, key=lambda x: abs(x - S))

rows = []
for strike in strikes:
    c  = bs(S, strike, T, r, sigma, "call")
    p  = bs(S, strike, T, r, sigma, "put")
    ci = max(0.0, S - strike)
    pi = max(0.0, strike - S)
    status = "ATM" if strike == atm_k else ("Call ITM" if strike < S else "Put ITM")
    rows.append({
        "Strike":         f"${strike:,.2f}",
        "Status":         status,
        "Call price":     f"${c:.4f}",
        "Call intrinsic": f"${ci:.2f}",
        "Call time val":  f"${c - ci:.4f}",
        "Put price":      f"${p:.4f}",
        "Put intrinsic":  f"${pi:.2f}",
        "Put time val":   f"${p - pi:.4f}",
    })

df = pd.DataFrame(rows)

def colour_row(row):
    s = row["Status"]
    if s == "ATM":      return ["background-color:#dbeafe"] * len(row)
    if s == "Call ITM": return ["background-color:#dcfce7"] * len(row)
    if s == "Put ITM":  return ["background-color:#fee2e2"] * len(row)
    return [""] * len(row)

st.dataframe(df.style.apply(colour_row, axis=1), use_container_width=True, hide_index=True)

st.divider()

# Charts
chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("Stock price — last 6 months")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        line=dict(color="rgb(59,130,246)", width=2), name="Close"
    ))
    fig1.add_hline(y=S, line_dash="dash", line_color="orange",
                   annotation_text=f"S = ${S:,.2f}", annotation_position="bottom right")
    fig1.add_hline(y=K, line_dash="dot", line_color="red",
                   annotation_text=f"K = ${K:,.2f}", annotation_position="top right")
    fig1.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", tickprefix="$"),
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

with chart2:
    st.subheader("P&L at expiry")
    spot_range   = np.linspace(S * 0.4, S * 1.6, 300)
    call_pnl     = [max(0, sp - K) - call_px for sp in spot_range]
    put_pnl      = [max(0, K - sp) - put_px  for sp in spot_range]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=spot_range, y=call_pnl, name="Call P&L",
                              line=dict(color="rgb(34,197,94)", width=2)))
    fig2.add_trace(go.Scatter(x=spot_range, y=put_pnl,  name="Put P&L",
                              line=dict(color="rgb(239,68,68)", width=2)))
    fig2.add_hline(y=0, line_color="gray", line_width=1)
    fig2.add_vline(x=float(K), line_dash="dash", line_color="orange",
                   annotation_text=f"K = ${K:,.2f}")
    fig2.add_vline(x=float(S), line_dash="dot", line_color="steelblue",
                   annotation_text=f"S = ${S:,.2f}")
    fig2.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title="Stock price at expiry", tickprefix="$"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", title="P&L ($)"),
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Black-Scholes model · Yahoo Finance · Educational purposes only · Not financial advice")
