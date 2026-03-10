from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_amount(df: pd.DataFrame) -> pd.Series:
    if "amount" in df.columns:
        amt = pd.to_numeric(df["amount"], errors="coerce")
    else:
        amt = pd.Series([np.nan] * len(df), index=df.index)
    if amt.isna().all() or float(amt.fillna(0).abs().sum()) <= 0.0:
        vol = pd.to_numeric(df.get("volume"), errors="coerce")
        close = pd.to_numeric(df.get("close"), errors="coerce")
        # Tencent volume is in hands (100 shares).
        amt = vol * close * 100.0
    return amt


def _rsi_wilder(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _safe_log_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    n = pd.to_numeric(numer, errors="coerce")
    d = pd.to_numeric(denom, errors="coerce")
    ratio = n / d
    ratio = ratio.where((n > 0) & (d > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(ratio)
    out = pd.Series(out, index=ratio.index)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


_CLEAN_DROP = {
    # Market-wide series (constant per date in cross-section)
    "market_ret_5d",
    "market_ret_20d",
    "market_trend_50",
    "market_trend_200",
    "market_regime",
    "market_vol_20",
    "market_atr_pct",
    # Redundant / weak volume factors
    "volume_change_5d",
    "volume_change_10d",
    "volume_change_20d",
    "volume_ratio_20",
    "volume_surge_5",
    "volume_surge_20",
    "volume_price_corr_20",
    "price_high_volume_low",
    "price_low_volume_high",
    # Money flow variants with weak signal
    "amount_ratio_20",
    "money_flow",
    "money_flow_20",
    "money_flow_ratio_20",
    "volume_per_price",
    # Redundant / noisy price range
    "high_low_range",
    "close_open_range",
    # MA redundancy
    "ma_10",
    "ma_30",
    "price_vs_ma_10",
    "price_vs_ma_30",
}

_COMPACT_KEEP = [
    "ret_1d",
    "price_accel_5",
    "price_accel_10",
    "price_vs_high_20",
    "price_vs_ma_5",
    "gap_1d",
    "body_ratio",
    "volume_ratio_5",
    "bull_volume",
    "bear_volume",
    "volume_change_20d_neg",
    "volume_shrink_20d",
    "rsi_6",
    "rsi_12",
    "kdj_j",
    "wr_14",
    "cci_20",
    "atr_percent",
]


class FeatureFactory:
    """Create technical features for A-share daily bars."""

    def __init__(self, feature_set: str = "legacy"):
        self.feature_set = str(feature_set or "legacy").strip().lower()
        self.return_windows = [1, 3, 5, 10, 20, 60]
        self.vol_windows = [5, 10, 20]
        self.ma_windows = [5, 10, 20, 30, 60]

    def _apply_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_set == "clean":
            return df.drop(columns=list(_CLEAN_DROP), errors="ignore")
        if self.feature_set == "compact":
            out = df.copy()
            if "volume_change_20d" in out.columns:
                out["volume_change_20d_neg"] = -out["volume_change_20d"]
            if "volume_change_20d_neg" not in out.columns:
                out["volume_change_20d_neg"] = np.nan
            if "volume_shrink_20d" not in out.columns:
                out["volume_shrink_20d"] = np.nan
            for col in _COMPACT_KEEP:
                if col not in out.columns:
                    out[col] = np.nan
            return out[_COMPACT_KEEP].copy()
        return df

    def create_all_features(self, df: pd.DataFrame, market_df: pd.DataFrame | None = None) -> pd.DataFrame:
        feats = []
        feats.append(self.create_price_features(df))
        feats.append(self.create_volume_features(df))
        feats.append(self.create_technical_features(df))
        feats.append(self.create_volatility_features(df))
        if market_df is not None and not market_df.empty:
            feats.append(self.create_market_features(df, market_df))
        feats.append(self.create_money_flow_features(df))
        out = pd.concat(feats, axis=1)
        return self._apply_feature_set(out)

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")
        eps = 1e-6

        for w in self.return_windows:
            feat[f"ret_{w}d"] = _safe_log_ratio(close, close.shift(w))

        feat["price_vs_high_20"] = close / high.rolling(20, min_periods=20).max()
        feat["price_vs_low_20"] = close / low.rolling(20, min_periods=20).min()
        feat["price_vs_open"] = close / open_ - 1
        feat["gap_1d"] = open_ / close.shift(1) - 1
        feat["body_ratio"] = (close - open_).abs() / (high - low + eps)

        feat["price_accel_5"] = feat["ret_5d"] - feat["ret_10d"]
        feat["price_accel_10"] = feat["ret_10d"] - feat["ret_20d"]

        if self.feature_set == "clean":
            feat["ret_2d"] = _safe_log_ratio(close, close.shift(2))
            feat["ret_7d"] = _safe_log_ratio(close, close.shift(7))
            feat["gap_1d"] = open_ / close.shift(1) - 1
            feat["close_pos"] = (close - low) / (high - low + eps)

        return feat

    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        volume = pd.to_numeric(df["volume"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")

        for w in self.vol_windows:
            feat[f"volume_change_{w}d"] = _safe_log_ratio(volume, volume.shift(w))

        feat["volume_ratio_5"] = volume / volume.rolling(5, min_periods=5).mean().shift(1)
        feat["volume_ratio_20"] = volume / volume.rolling(20, min_periods=20).mean().shift(1)

        feat["volume_surge_5"] = (feat["volume_ratio_5"] > 1.5).astype(float)
        feat["volume_surge_20"] = (feat["volume_ratio_20"] > 2.0).astype(float)

        # Directional volume: treat "volume + up day" as positive, "volume + down day" as negative.
        feat["bull_volume"] = ((feat["volume_ratio_20"] > 1.2) & (close > open_)).astype(float)
        feat["bear_volume"] = ((feat["volume_ratio_20"] > 1.5) & (close < open_)).astype(float)

        feat["price_high_volume_low"] = (
            (close > close.rolling(20, min_periods=20).max().shift(1))
            & (volume < volume.rolling(20, min_periods=20).mean())
        ).astype(float)

        feat["price_low_volume_high"] = (
            (close < close.rolling(20, min_periods=20).min().shift(1))
            & (feat["volume_ratio_20"] > 1.5)
        ).astype(float)

        feat["volume_price_corr_20"] = close.rolling(20, min_periods=20).corr(volume)

        # Convert negative-IC volume trend into a "shrink" signal (20d volume down >10%).
        shrink_thr = np.log(0.9)
        feat["volume_shrink_20d"] = (feat["volume_change_20d"] < shrink_thr).astype(float)
        return feat

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        close = pd.to_numeric(df["close"], errors="coerce")

        ma_store: dict[int, pd.Series] = {}
        for w in self.ma_windows:
            ma = close.rolling(w, min_periods=w).mean()
            feat[f"ma_{w}"] = ma
            feat[f"price_vs_ma_{w}"] = close / ma - 1
            ma_store[w] = ma

        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        feat["bb_width"] = (bb_upper - bb_lower) / bb_mid
        eps = 1e-6
        feat["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + eps)

        feat["rsi_14"] = _rsi_wilder(close, 14)
        feat["rsi_6"] = _rsi_wilder(close, 6)
        feat["rsi_12"] = _rsi_wilder(close, 12)

        # KDJ (9,3,3)
        low9 = pd.to_numeric(df["low"], errors="coerce").rolling(9, min_periods=9).min()
        high9 = pd.to_numeric(df["high"], errors="coerce").rolling(9, min_periods=9).max()
        eps = 1e-6
        rsv = ((close - low9) / (high9 - low9 + eps) * 100.0).clip(0, 100)
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        feat["kdj_j"] = 3.0 * k - 2.0 * d

        # Williams %R (14)
        low14 = pd.to_numeric(df["low"], errors="coerce").rolling(14, min_periods=14).min()
        high14 = pd.to_numeric(df["high"], errors="coerce").rolling(14, min_periods=14).max()
        feat["wr_14"] = ((high14 - close) / (high14 - low14 + eps) * 100.0).clip(0, 100)

        # CCI (20)
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        tp = (high + low + close) / 3.0
        tp_ma = tp.rolling(20, min_periods=20).mean()
        md = (tp - tp_ma).abs().rolling(20, min_periods=20).mean()
        feat["cci_20"] = (tp - tp_ma) / (0.015 * md + eps)

        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        feat["macd"] = macd
        feat["macd_signal"] = macd_signal
        feat["macd_hist"] = macd - macd_signal

        if self.feature_set == "clean":
            feat["rsi_6"] = _rsi_wilder(close, 6)
            feat["rsi_24"] = _rsi_wilder(close, 24)
            ma5 = ma_store.get(5)
            ma20 = ma_store.get(20)
            ma60 = ma_store.get(60)
            if ma5 is not None and ma20 is not None:
                feat["ma_gap_5_20"] = ma5 / ma20 - 1
                feat["ma_5_slope"] = ma5 / ma5.shift(5) - 1
                feat["ma_20_slope"] = ma20 / ma20.shift(5) - 1
            if ma20 is not None and ma60 is not None:
                feat["ma_gap_20_60"] = ma20 / ma60 - 1
        return feat

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")

        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        feat["atr_14"] = tr.rolling(14, min_periods=14).mean()
        feat["atr_percent"] = feat["atr_14"] / close

        vol_map: dict[int, pd.Series] = {}
        for w in self.vol_windows:
            vol = close.pct_change().rolling(w, min_periods=w).std()
            feat[f"volatility_{w}d"] = vol
            vol_map[w] = vol

        if self.feature_set == "clean":
            vol60 = close.pct_change().rolling(60, min_periods=60).std()
            feat["volatility_60d"] = vol60
            if 20 in vol_map:
                feat["volatility_ratio_20_60"] = vol_map[20] / vol60

        feat["high_low_range"] = (high - low) / close
        feat["close_open_range"] = (close - open_) / open_
        return feat

    def create_market_features(self, df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        if "date" not in df.columns:
            return feat

        m = market_df.copy()
        if "date" not in m.columns:
            return feat
        m = m.sort_values("date")
        m_close = pd.to_numeric(m["close"], errors="coerce")
        m_high = pd.to_numeric(m.get("high"), errors="coerce")
        m_low = pd.to_numeric(m.get("low"), errors="coerce")

        m["market_ret_5d"] = m_close.pct_change(5)
        m["market_ret_20d"] = m_close.pct_change(20)
        m["market_ma50"] = m_close.rolling(50, min_periods=50).mean()
        m["market_ma200"] = m_close.rolling(200, min_periods=200).mean()
        m["market_trend_50"] = m_close / m["market_ma50"] - 1
        m["market_trend_200"] = m_close / m["market_ma200"] - 1
        m["market_regime"] = (m_close >= m["market_ma200"]).astype(float)
        m["market_vol_20"] = m_close.pct_change().rolling(20, min_periods=20).std()

        if m_high.notna().any() and m_low.notna().any():
            high_low = m_high - m_low
            high_close = (m_high - m_close.shift()).abs()
            low_close = (m_low - m_close.shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            m["market_atr_14"] = tr.rolling(14, min_periods=14).mean()
            m["market_atr_pct"] = m["market_atr_14"] / m_close

        date_col = df["date"].astype(str)
        for col in [
            "market_ret_5d",
            "market_ret_20d",
            "market_trend_50",
            "market_trend_200",
            "market_regime",
            "market_vol_20",
            "market_atr_pct",
        ]:
            if col in m.columns:
                feat[col] = date_col.map({str(r["date"]): float(r[col]) for _, r in m.iterrows()})

        close = pd.to_numeric(df["close"], errors="coerce")
        feat["rel_strength_5d"] = close.pct_change(5) - feat.get("market_ret_5d")
        feat["rel_strength_20d"] = close.pct_change(20) - feat.get("market_ret_20d")
        return feat

    def create_money_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        amount = _ensure_amount(df)
        close = pd.to_numeric(df["close"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")

        feat["amount_ratio_5"] = amount / amount.rolling(5, min_periods=5).mean().shift(1)
        avg_amount_20 = amount.rolling(20, min_periods=20).mean()
        feat["amount_ratio_20"] = amount / avg_amount_20.shift(1)
        feat["avg_amount_20"] = avg_amount_20

        direction = np.where(close > open_, 1.0, np.where(close < open_, -1.0, 0.0))
        money_flow = amount * direction
        feat["money_flow"] = money_flow
        feat["money_flow_20"] = money_flow.rolling(20, min_periods=20).sum()
        feat["money_flow_ratio_20"] = money_flow.rolling(20, min_periods=20).mean() / amount.rolling(20, min_periods=20).mean()
        feat["volume_per_price"] = pd.to_numeric(df["volume"], errors="coerce") / pd.to_numeric(df["close"], errors="coerce")
        return feat
