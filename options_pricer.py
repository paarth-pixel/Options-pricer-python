import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import date, timedelta

st.set_page_config(page_title="Stock Options Pricer", layout="wide", page_icon="📈")

# ── Black-Scholes ──────────────────────────────────────
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

# ── Digital / At-Expiry Knockout ───────────────────────
def bs_digital(S, K, T, r, sig, kind="call"):
    """Cash-or-nothing digital: pays $1 if condition met at T, else $0."""
    if T <= 0:
        if kind == "call":
            return 1.0 if S > K else 0.0
        return 1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if kind == "call":
        return np.exp(-r * T) * norm.cdf(d2)
    return np.exp(-r * T) * norm.cdf(-d2)

def at_expiry_ko(S, K, B, T, r, sig, kind="call"):
    """
    Vanilla call/put that is knocked out (pays $0) if the barrier B is
    breached ONLY at expiry (not monitored along the path). Because the
    barrier check is a single terminal condition, this is not actually
    path-dependent -- it can be replicated exactly with vanilla options:

    Call (B above K):  [Call(K) - Call(B)]  -  (B-K) * DigitalCall(B)
    Put  (B below K):  [Put(K)  - Put(B)]   -  (K-B) * DigitalPut(B)

    The spread gives the normal ramp; the digital leg cancels the flat
    region that would otherwise remain once the barrier is crossed.
    """
    if kind == "call":
        if T <= 0:
            return 0.0 if S > B else max(0.0, S - K)
        spread = bs(S, K, T, r, sig, "call") - bs(S, B, T, r, sig, "call")
        cap_removal = (B - K) * bs_digital(S, B, T, r, sig, "call")
        return max(spread - cap_removal, 0.0)
    else:
        if T <= 0:
            return 0.0 if S < B else max(0.0, K - S)
        spread = bs(S, K, T, r, sig, "put") - bs(S, B, T, r, sig, "put")
        cap_removal = (K - B) * bs_digital(S, B, T, r, sig, "put")
        return max(spread - cap_removal, 0.0)

# ── Data fetching ──────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_stock(ticker):
    try:
        t = yf.Ticker(ticker.upper())
        hist = t.history(period="6mo")
        if hist.empty:
            return None, None, None, None, None
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

def _strike_interp_iv(chain_calls, S):
    """Linearly interpolate ATM IV between the two listed strikes bracketing S."""
    calls = chain_calls[chain_calls["impliedVolatility"] > 0].sort_values("strike")
    if calls.empty:
        return None
    strikes = calls["strike"].to_numpy()
    ivs = calls["impliedVolatility"].to_numpy()
    if S <= strikes[0]:
        return float(ivs[0])
    if S >= strikes[-1]:
        return float(ivs[-1])
    idx = int(np.searchsorted(strikes, S))
    k_lo, k_hi = strikes[idx - 1], strikes[idx]
    iv_lo, iv_hi = ivs[idx - 1], ivs[idx]
    if k_hi == k_lo:
        return float(iv_lo)
    w = (S - k_lo) / (k_hi - k_lo)
    return float(iv_lo + w * (iv_hi - iv_lo))

@st.cache_data(ttl=300)
def fetch_iv(ticker, S, target_date_str):
    """
    ATM implied vol for the target expiry, interpolated on TWO axes:
      1. Strike  - linear interpolation between the two listed strikes
                   bracketing spot (instead of snapping to nearest strike).
      2. Expiry  - linear interpolation in TOTAL VARIANCE (sigma^2 * T)
                   between the nearest listed expiry before and after the
                   target date. Variance (not vol) is the quantity that
                   scales linearly with time under Black-Scholes, so this
                   is the standard way term-structure interpolation is done
                   on a vol surface.
    Falls back gracefully to a single expiry if the target date is outside
    the range of listed expiries (can't interpolate, only extrapolate).
    """
    try:
        t = yf.Ticker(ticker.upper())
        exps = t.options
        if not exps:
            return None, None

        target = date.fromisoformat(target_date_str)
        today = date.today()
        exp_dates = [date.fromisoformat(e) for e in exps]

        before = [(e, d) for e, d in zip(exps, exp_dates) if d <= target]
        after  = [(e, d) for e, d in zip(exps, exp_dates) if d >= target]

        if before and after:
            e1, d1 = max(before, key=lambda x: x[1])   # nearest expiry <= target
            e2, d2 = min(after,  key=lambda x: x[1])   # nearest expiry >= target
        else:
            # target is outside the listed range - no bracket available,
            # just use whichever single expiry is closest
            e1, d1 = min(zip(exps, exp_dates), key=lambda x: abs((x[1] - target).days))
            e2, d2 = e1, d1

        def atm_iv(exp):
            chain = t.option_chain(exp)
            return _strike_interp_iv(chain.calls, S)

        iv1 = atm_iv(e1)
        iv2 = atm_iv(e2) if e2 != e1 else iv1
        if iv1 is None and iv2 is None:
            return None, None
        iv1 = iv1 if iv1 is not None else iv2
        iv2 = iv2 if iv2 is not None else iv1

        T1 = max((d1 - today).days, 1) / 365
        T2 = max((d2 - today).days, 1) / 365
        Tt = max((target - today).days, 1) / 365

        if e1 == e2 or T1 == T2:
            iv_final = iv1
            label = e1
        else:
            var1, var2 = iv1**2 * T1, iv2**2 * T2
            w = min(max((Tt - T1) / (T2 - T1), 0.0), 1.0)
            var_t = var1 + w * (var2 - var1)
            iv_final = np.sqrt(max(var_t, 0.0) / Tt)
            label = f"{e1} → {e2} (interpolated)"

        if iv_final and iv_final > 0:
            return round(iv_final * 100, 1), label
        return None, None
    except Exception:
        return None, None

@st.cache_data(ttl=3600)
def resolve_ticker(query):
    query = query.strip()
    if not query:
        return None, None
    try:
        url = (
            "https://query2.finance.yahoo.com/v1/finance/search"
            f"?q={requests.utils.quote(query)}&quotesCount=6&newsCount=0&listsCount=0"
        )
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return query.upper(), None
        quotes = resp.json().get("quotes", [])
        equities = [q for q in quotes if q.get("quoteType") in ("EQUITY", "ETF")]
        if not equities:
            return query.upper(), None
        best   = equities[0]
        symbol = best.get("symbol", query.upper())
        name   = best.get("longname") or best.get("shortname") or symbol
        return symbol, name
    except Exception:
        return query.upper(), None

# ── Helpers ────────────────────────────────────────────
def strike_step(price):
    if price < 5:    return 0.05
    if price < 20:   return 0.5
    if price < 50:   return 1.0
    if price < 200:  return 2.5
    if price < 500:  return 5.0
    if price < 1000: return 10.0
    return 25.0

def gen_strikes(S, n=12):
    step = strike_step(S)
    base = round(round(S / step) * step, 4)
    return sorted(set(round(base + step * i, 4) for i in range(-5, n - 5) if base + step * i > 0))

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Options Pricer")
    st.caption("Live Black-Scholes for any stock")

    ticker_input = st.text_input(
        "Search stock", value="SPCX",
        help="Type company name (Tesla) or ticker (TSLA)"
    ).strip()

    resolved_ticker, resolved_name = resolve_ticker(ticker_input)
    if resolved_ticker and resolved_ticker.upper() != ticker_input.upper():
        st.caption(f"Matched: **{resolved_ticker}** — {resolved_name or ''}")
    ticker = resolved_ticker or ticker_input.upper()

    result = fetch_stock(ticker)
    if result[0] is None:
        st.error(f"Could not fetch '{ticker}' — try a different name or ticker.")
        st.stop()

    live_price, prev_price, hist, hvol, company_name = result
    daily_chg = round(live_price - prev_price, 2)
    daily_pct = round(daily_chg / prev_price * 100, 2) if prev_price else 0

    if daily_chg >= 0:
        st.success(f"**${live_price:,.2f}**  ▲ +${daily_chg} (+{daily_pct}%) today")
    else:
        st.error(f"**${live_price:,.2f}**  ▼ ${daily_chg} ({daily_pct}%) today")

    st.caption(f"30-day realised vol: **{hvol}%**  ·  Updates every 60s")
    st.divider()

    st.subheader("Black-Scholes inputs")

    step  = strike_step(live_price)

    # S and K — number inputs
    S = st.number_input(
        "S — Stock price ($)",
        min_value=0.01, value=float(live_price), step=float(step), format="%.2f",
        help="Current stock price. Edit to model scenarios."
    )
    K = st.number_input(
        "K — Strike price ($)",
        min_value=0.01, value=float(round(round(live_price / step) * step, 2)),
        step=float(step), format="%.2f",
        help="The fixed price written into the option contract."
    )

    # Expiry date — calendar picker
    default_expiry = date.today() + timedelta(days=180)
    expiry_date = st.date_input(
        "Expiry date",
        value=default_expiry,
        min_value=date.today() + timedelta(days=1),
        help="Pick any expiry date — no limit on how far out."
    )
    T_days = (expiry_date - date.today()).days
    T = max(T_days, 1) / 365
    st.caption(f"**{T_days} days** to expiry  ({T:.4f} years)")

    # r — number input
    r_pct = st.number_input(
        "r — Risk-free rate (%)",
        min_value=0.0, max_value=50.0, value=4.0, step=0.25, format="%.2f",
        help="US Treasury yield used as the risk-free rate."
    )
    r = r_pct / 100

    # σ — fetched from options chain, shown as editable number input
    market_iv, used_exp = fetch_iv(ticker, S, expiry_date.isoformat())
    if market_iv:
        st.caption(f"Market IV fetched from options chain expiry **{used_exp}**")
        default_sig = market_iv
        iv_source = f"Market IV ({used_exp})"
    else:
        default_sig = max(10.0, float(hvol)) if hvol else 30.0
        iv_source = "30-day historical vol (no options chain available)"
        st.caption(f"ℹ️ No options chain found — using historical vol as default.")

    sig_pct = st.number_input(
        "σ — Implied volatility (%)",
        min_value=0.1, max_value=1000.0, value=float(default_sig), step=0.5, format="%.1f",
        help=f"Auto-filled from: {iv_source}. Edit to override."
    )
    sigma = sig_pct / 100

    st.divider()
    st.caption("All values update instantly as you edit any field.")

# ── Main ───────────────────────────────────────────────
st.header(f"{company_name}  ·  {ticker}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("S  (stock price)", f"${S:,.2f}")
c2.metric("K  (strike)",      f"${K:,.2f}")
c3.metric("Expiry",           str(expiry_date))
c4.metric("r  (rate)",        f"{r_pct:.2f}%")
c5.metric("σ  (vol)",         f"{sig_pct:.1f}%")
c6.metric("T  (years)",       f"{T:.4f}")

st.divider()

call_px = bs(S, K, T, r, sigma, "call")
put_px  = bs(S, K, T, r, sigma, "put")
c_int   = max(0.0, S - K)
p_int   = max(0.0, K - S)

left, right = st.columns(2)
left.metric("Call price at K",  f"${call_px:.2f}",
            delta=f"Intrinsic ${c_int:.2f}  +  Time ${call_px - c_int:.2f}")
right.metric("Put price at K",  f"${put_px:.2f}",
             delta=f"Intrinsic ${p_int:.2f}  +  Time ${put_px - p_int:.2f}")

g = greeks(S, K, T, r, sigma)
if g:
    with st.expander("Greeks for selected K", expanded=True):
        gcols = st.columns(4)
        for i, (label, val) in enumerate(g.items()):
            gcols[i % 4].metric(label, val)

st.divider()

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
        "Call price":     f"${c:.2f}",
        "Call intrinsic": f"${ci:.2f}",
        "Call time val":  f"${c - ci:.2f}",
        "Put price":      f"${p:.2f}",
        "Put intrinsic":  f"${pi:.2f}",
        "Put time val":   f"${p - pi:.2f}",
    })

df = pd.DataFrame(rows)

def colour_row(row):
    s = row["Status"]
    if s == "ATM":
        return ["background-color:#1e3a5f; color:#93c5fd; font-weight:600"] * len(row)
    if s == "Call ITM":
        return ["background-color:#14532d; color:#86efac"] * len(row)
    if s == "Put ITM":
        return ["background-color:#7f1d1d; color:#fca5a5"] * len(row)
    return ["background-color:#1e293b; color:#cbd5e1"] * len(row)

st.dataframe(df.style.apply(colour_row, axis=1), use_container_width=True, hide_index=True)

st.divider()

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
    spot_range = np.linspace(S * 0.4, S * 1.6, 300)
    call_pnl   = [max(0, sp - K) - call_px for sp in spot_range]
    put_pnl    = [max(0, K - sp) - put_px  for sp in spot_range]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=spot_range, y=call_pnl, name="Call P&L",
                              line=dict(color="rgb(34,197,94)", width=2)))
    fig2.add_trace(go.Scatter(x=spot_range, y=put_pnl, name="Put P&L",
                              line=dict(color="rgb(239,68,68)", width=2)))
    fig2.add_hline(y=0, line_color="gray", line_width=1)
    fig2.add_vline(x=float(K), line_dash="dash", line_color="orange",
                   annotation_text=f"K=${K:,.2f}")
    fig2.add_vline(x=float(S), line_dash="dot", line_color="steelblue",
                   annotation_text=f"S=${S:,.2f}")
    fig2.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title="Stock price at expiry", tickprefix="$"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", title="P&L ($)"),
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("At-Expiry Knockout Spread")
st.caption("Barrier checked only on the expiry date, so this is priced exactly with vanilla options — no simulation needed.")

ko1, ko2, ko3 = st.columns([1, 1, 2])
with ko1:
    ko_kind = st.selectbox("Type", ["call", "put"], index=0)
with ko2:
    default_B = K + step * 10 if ko_kind == "call" else max(step, K - step * 10)
    B = st.number_input(
        f"B — Barrier ($) — {'above' if ko_kind=='call' else 'below'} K",
        min_value=0.01, value=float(default_B), step=float(step), format="%.2f",
        help="Option pays $0 if the stock finishes past this level on expiry day."
    )
with ko3:
    ko_price = at_expiry_ko(S, K, B, T, r, sigma, ko_kind)
    vanilla_price = bs(S, K, T, r, sigma, ko_kind)
    st.metric(
        f"KO {ko_kind} price  (K={K:.2f}, B={B:.2f})",
        f"${ko_price:.2f}",
        delta=f"vs ${vanilla_price:.2f} vanilla  ·  {ko_price - vanilla_price:+.2f} from KO risk"
    )

kc1, kc2 = st.columns(2)

with kc1:
    st.caption("Payoff at expiry — the knockout cliff")
    lo, hi = (K * 0.4, B * 1.5) if ko_kind == "call" else (B * 0.5, K * 1.6)
    spot_range_ko = np.linspace(lo, hi, 400)
    if ko_kind == "call":
        payoff = [ (sp - K) if (K < sp <= B) else 0.0 for sp in spot_range_ko ]
        vanilla_payoff = [ max(0, sp - K) for sp in spot_range_ko ]
    else:
        payoff = [ (K - sp) if (B <= sp < K) else 0.0 for sp in spot_range_ko ]
        vanilla_payoff = [ max(0, K - sp) for sp in spot_range_ko ]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=spot_range_ko, y=vanilla_payoff, name="Vanilla payoff",
                               line=dict(color="rgba(148,163,184,0.6)", width=2, dash="dot")))
    fig3.add_trace(go.Scatter(x=spot_range_ko, y=payoff, name="KO payoff",
                               line=dict(color="rgb(239,68,68)", width=3)))
    fig3.add_vline(x=float(K), line_dash="dash", line_color="orange", annotation_text=f"K={K:.2f}")
    fig3.add_vline(x=float(B), line_dash="dash", line_color="red", annotation_text=f"B={B:.2f}")
    fig3.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title="Stock price at expiry", tickprefix="$"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", title="Payoff ($)"),
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig3, use_container_width=True)

with kc2:
    st.caption("Value today vs spot — how the KO price behaves before expiry")
    spot_range_val = np.linspace(lo, hi, 200)
    ko_values = [at_expiry_ko(sp, K, B, T, r, sigma, ko_kind) for sp in spot_range_val]
    vanilla_values = [bs(sp, K, T, r, sigma, ko_kind) for sp in spot_range_val]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=spot_range_val, y=vanilla_values, name="Vanilla value",
                               line=dict(color="rgba(148,163,184,0.6)", width=2, dash="dot")))
    fig4.add_trace(go.Scatter(x=spot_range_val, y=ko_values, name="KO value",
                               line=dict(color="rgb(59,130,246)", width=3)))
    fig4.add_vline(x=float(S), line_dash="dot", line_color="steelblue", annotation_text=f"S={S:.2f}")
    fig4.add_vline(x=float(B), line_dash="dash", line_color="red", annotation_text=f"B={B:.2f}")
    fig4.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title="Stock price today", tickprefix="$"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", title="Option value ($)"),
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig4, use_container_width=True)

st.caption("Black-Scholes model · Yahoo Finance · Educational purposes only · Not financial advice")
