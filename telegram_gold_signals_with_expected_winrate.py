#!/usr/bin/env python3
# Telegram Gold Signals Bot - Expected Winrate per trade
# File: telegram_gold_signals_with_expected_winrate.py
#
# Sends XAU/USD signals every 15 minutes (MACD + RSI on 15m) and computes
# a per-trade expected winrate based on MACD, RSI, ATR and EMA50(1H) trend.
# Message format matches user's requested layout.

import os, io, time, json, threading, logging, traceback
from datetime import datetime, timezone
import numpy as np, pandas as pd, yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# --- CONFIG ---
BOT_TOKEN = os.getenv("BOT_TOKEN") or "PUT_YOUR_TOKEN_HERE"
SYMBOL = "GC=F"            # yfinance symbol for gold futures (XAU/USD)
PERIOD = "14d"
PERIOD_1H = "60d"
INTERVAL = "15m"
INTERVAL_1H = "1h"
CHECK_INTERVAL_SECONDS = 15 * 60   # 15 minutes
SUB_FILE = "subscribers.json"
SIGNAL_LOG = "signals_log.json"
CONTACT_LINE = "\n━━━━━━━━━━━━━━━━━━━━━━━\n📞 للخدمات المدفوعة والمساعدة: @fahdcryptoo"

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# --- persistence ---
def load_subscribers():
    try:
        if os.path.exists(SUB_FILE):
            with open(SUB_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
    except Exception:
        logger.exception("load_subscribers failed")
    return set()

def save_subscribers(subs):
    try:
        with open(SUB_FILE, "w", encoding="utf-8") as f:
            json.dump(list(subs), f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("save_subscribers failed")

def append_signal_log(entry):
    try:
        logs = []
        if os.path.exists(SIGNAL_LOG):
            with open(SIGNAL_LOG, "r", encoding="utf-8") as f:
                logs = json.load(f)
        logs.append(entry)
        with open(SIGNAL_LOG, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("append_signal_log failed")

subscribers = load_subscribers()

# --- helpers ---
def safe_float(x):
    try:
        if isinstance(x, pd.Series): return float(x.iloc[0])
        if hasattr(x, "item"): return float(x.item())
        return float(x)
    except Exception:
        return float("nan")

def fetch_yf(symbol, period=None, interval="15m", start=None, end=None):
    try:
        if start is not None and end is not None:
            df = yf.download(tickers=symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
        else:
            df = yf.download(tickers=symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        return df.dropna()
    except Exception:
        logger.exception("fetch_yf error for %s", symbol)
        return None

def compute_basic(df):
    df = df.copy()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta = df["Close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain, avg_loss = gain.rolling(14).mean(), loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def ema50_trend_1h():
    df1h = fetch_yf(SYMBOL, period=PERIOD_1H, interval=INTERVAL_1H)
    if df1h is None:
        return None
    ema50 = df1h["Close"].ewm(span=50, adjust=False).mean()
    try:
        last = float(ema50.iloc[-1]); prev = float(ema50.iloc[-2])
        if last > prev: return "UP"
        if last < prev: return "DOWN"
        return "FLAT"
    except Exception:
        return None

def build_chart(df, entry=None, tp=None, sl=None, side=None):
    try:
        plt.ioff()
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(df.index, df["Close"], lw=1.2, label="Price")
        if entry is not None: ax.axhline(entry, ls="--", label=f"Entry {entry:.2f}")
        if tp is not None: ax.axhline(tp, ls="--", label=f"TP {tp:.2f}")
        if sl is not None: ax.axhline(sl, ls="--", label=f"SL {sl:.2f}")
        if side == "BUY":
            ax.set_facecolor((1.0, 0.99, 0.98))
        elif side == "SELL":
            ax.set_facecolor((0.98, 0.99, 1.0))
        ax.legend(loc="best"); ax.set_title("XAU/USD 15m")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig); buf.seek(0)
        return buf
    except Exception:
        logger.exception("build_chart failed")
        return None

# --- expected winrate calculator ---
def expected_winrate(macd_now, macd_sig, rsi_now, atr, entry, trend1h):
    """
    Compute a heuristic expected winrate (%) from 4 factors with weights:
    - MACD momentum (0-30%)
    - RSI proximity to neutral 50 (0-25%)
    - ATR relative (lower ATR => higher weight) (0-20%)
    - EMA50(1H) trend alignment (0-25%)
    Final expected = base + weighted contributions (clipped 40..95 for realism)
    """
    base = 45.0  # conservative base probability in percent

    # MACD score: stronger separation of MACD from signal increases score
    macd_diff = macd_now - macd_sig
    macd_score = 0.0
    try:
        macd_strength = abs(macd_diff)  # magnitude
        # scale roughly: small diff -> small score, larger diff -> larger score
        macd_score = min(30.0, 30.0 * (min(macd_strength / (0.5 if entry>0 else 1.0), 1.0)))
    except Exception:
        macd_score = 0.0

    # RSI score: closeness to 50 is better (not overbought/oversold)
    rsi_score = 0.0
    try:
        dist = abs(rsi_now - 50.0)
        # dist 0 -> full score 25, dist 25 -> zero
        rsi_score = max(0.0, 25.0 * max(0.0, (1 - dist / 25.0)))
    except Exception:
        rsi_score = 0.0

    # ATR score: lower ATR (less noise) gives higher score. Compare atr to entry as pct.
    atr_score = 0.0
    try:
        pct = atr / entry if entry and entry>0 else 0.01
        # if pct small (<0.002) => strong score; if pct large (>0.01) => weak
        ratio = max(0.0, min(1.0, (0.01 - pct) / 0.009))  # maps pct in [0.001..0.01] -> 1..0
        atr_score = 20.0 * ratio
    except Exception:
        atr_score = 0.0

    # EMA50 1H trend alignment
    trend_score = 0.0
    try:
        if trend1h is None:
            trend_score = 10.0  # partial if unknown
        else:
            if macd_now > macd_sig and trend1h == "UP":
                trend_score = 25.0
            elif macd_now < macd_sig and trend1h == "DOWN":
                trend_score = 25.0
            elif trend1h == "FLAT":
                trend_score = 10.0
            else:
                trend_score = 0.0
    except Exception:
        trend_score = 0.0

    expected = base + macd_score + rsi_score + atr_score + trend_score
    # clip to realistic bounds
    expected = max(40.0, min(95.0, expected))
    return round(expected, 1)

# --- generate signal ---
def generate_signal():
    df = fetch_yf(SYMBOL, period=PERIOD, interval=INTERVAL)
    if df is None:
        return None, None, None
    df = compute_basic(df)
    if len(df) < 5:
        return None, None, None
    last = df.iloc[-1]
    macd_now = safe_float(last.get("MACD", float("nan")))
    macd_sig = safe_float(last.get("MACD_signal", float("nan")))
    rsi_now = safe_float(last.get("RSI", float("nan")))
    entry = safe_float(last.get("Close", float("nan")))

    try:
        atr = float((df["High"].rolling(14).max() - df["Low"].rolling(14).min()).iloc[-1])
    except Exception:
        atr = float("nan")
    if np.isnan(atr) or atr == 0:
        atr = max(0.5, safe_float(df["Close"].pct_change().std() * entry))

    trend1h = ema50_trend_1h()

    # signal decision (MACD + RSI)
    side = "HOLD"
    if macd_now > macd_sig and rsi_now < 70:
        side = "BUY"
    elif macd_now < macd_sig and rsi_now > 30:
        side = "SELL"
    else:
        side = "HOLD"

    tp = sl = None
    if side == "BUY":
        sl = entry - atr; tp = entry + 2 * atr
    elif side == "SELL":
        sl = entry + atr; tp = entry - 2 * atr

    # compute expected winrate for this trade
    exp_wr = expected_winrate(macd_now, macd_sig, rsi_now, atr, entry, trend1h)

    # build message (requested format)
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "⚡️ إشـارة تـداول الـذهـب ⚡️",
        "━━━━━━━━━━━━━━━━━━━━━━━",
        "💎 الزوج: XAU/USD",
        f"📅 الوقت: {now_str}",
        "",
        "🎯 نقاط الصفقة:",
        f"🟢 دخول: {entry:.2f}" if not np.isnan(entry) else "🟢 دخول: N/A",
        f"🟠 هدف: {tp:.2f}" if tp is not None else "🟠 هدف: N/A",
        f"🔴 وقف: {sl:.2f}" if sl is not None else "🔴 وقف: N/A",
        "",
        f"✅ نسبة النجاح المتوقعة: {exp_wr:.1f}%",
        "⚠️ ملاحظات: إشارات آلية. استخدم إدارة مخاطر مناسبة.",
        "━━━━━━━━━━━━━━━━━━━━━━━",
        f"🔑 التوصية: {'🟩 شراء (BUY)' if side=='BUY' else ('🟥 بيع (SELL)' if side=='SELL' else '⚪ انتظار')}"
    ]
    text = "\n".join(lines) + CONTACT_LINE
    chart = build_chart(df.tail(200), entry, tp, sl, side)

    record = {
        "timestamp_utc": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "side": side,
        "entry": float(entry) if not np.isnan(entry) else None,
        "tp": float(tp) if tp is not None else None,
        "sl": float(sl) if sl is not None else None,
        "expected_winrate": exp_wr
    }
    return text, chart, record

# --- send functions ---
def send_to_subscribers(text, chart, rec):
    sent = 0
    for uid in list(subscribers):
        try:
            if chart:
                requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                              data={"chat_id": uid, "caption": text, "disable_notification": False},
                              files={"photo": ("chart.png", chart.getvalue(), "image/png")}, timeout=10)
            else:
                requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                              json={"chat_id": uid, "text": text, "disable_notification": False}, timeout=8)
            sent += 1
        except Exception:
            logger.exception("Failed to send to %s", uid)
    rec2 = rec.copy(); rec2["sent_to"] = sent
    append_signal_log(rec2)
    return sent

# --- handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    welcome = "✅ تم اشتراكك بنجاح وستتلقى إشارات الذهب تلقائيًا كل 15 دقيقة." + CONTACT_LINE
    if chat_id not in subscribers:
        subscribers.add(chat_id); save_subscribers(subscribers)
        try:
            text, chart, rec = generate_signal()
            if text and chart:
                await context.bot.send_photo(chat_id=chat_id, photo=chart, caption=welcome + "\n\n" + text)
            else:
                await context.bot.send_message(chat_id=chat_id, text=welcome)
        except Exception:
            logger.exception("start: send failed")
            await context.bot.send_message(chat_id=chat_id, text=welcome)
    else:
        await update.message.reply_text("أنت مشترك فعلاً ✅" + CONTACT_LINE)

async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in subscribers:
        subscribers.remove(chat_id); save_subscribers(subscribers)
        await update.message.reply_text("🗑️ تم إلغاء اشتراكك." + CONTACT_LINE)
    else:
        await update.message.reply_text("أنت غير مشترك حالياً." + CONTACT_LINE)

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text, chart, rec = generate_signal()
        if not text:
            await update.message.reply_text("⚠️ لا توجد بيانات كافية الآن." + CONTACT_LINE)
            return
        if chart:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=chart, caption=text)
        else:
            await update.message.reply_text(text)
    except Exception:
        logger.exception("signal command failed")
        await update.message.reply_text("حدث خطأ أثناء تجهيز الإشارة." + CONTACT_LINE)

# --- background loop ---
def background_loop():
    last_key = None
    while True:
        try:
            text, chart, rec = generate_signal()
            if text and rec and rec.get("side") not in (None, "HOLD"):
                key = f"{rec['timestamp_utc']}_{rec.get('side')}_{rec.get('entry')}"
                if key != last_key:
                    last_key = key
                    cnt = send_to_subscribers(text, chart, rec)
                    logger.info("Background: sent %s signal to %d subscribers (expected %.1f%%)",
                                rec.get("side"), cnt, rec.get("expected_winrate", 0.0))
            else:
                logger.info("Background: no actionable signal or HOLD")
        except Exception:
            logger.exception("background loop error: %s", traceback.format_exc())
        time.sleep(CHECK_INTERVAL_SECONDS)

def main():
    if not BOT_TOKEN or "PUT_YOUR_TOKEN_HERE" in BOT_TOKEN:
        print("❌ ضع رمز البوت الصحيح أولاً.")
        return
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("signal", signal_command))
    threading.Thread(target=background_loop, daemon=True).start()
    logger.info("🤖 Bot started (run_polling)")
    app.run_polling()

if __name__ == "__main__":
    main()
