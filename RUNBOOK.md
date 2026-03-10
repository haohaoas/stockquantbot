# StockQuantBot Runbook

## 1) Single Source Of Truth
- Online scoring model path: `config/default.yaml` -> `signals.model_ref.path`
- Backtest scripts (`backtest_model.py`, `backtest_rule_model.py`) now default to the same config path.
- If needed, you can still override by passing `--model`.

## 2) Start Services
- Backend (API):
```bash
cd /Users/haohao/stockquantbot
/opt/homebrew/anaconda3/envs/fixed_env/bin/python -m uvicorn api:app --reload --port 8000
```
- Frontend (Vite):
```bash
cd /Users/haohao/stockquantbot/web
npm install
npm run dev
```

## 3) Daily Data Maintenance
- Update A-share daily history:
```bash
cd /Users/haohao/stockquantbot
HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
/opt/homebrew/anaconda3/envs/fixed_env/bin/python update_manual_hist.py \
  --universe-file ./data/universe.csv \
  --only-stale --retry-direct --fill-rt --use-proxy \
  --workers 4 --sleep 0.05
```
- Update index history:
```bash
cd /Users/haohao/stockquantbot
HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
/opt/homebrew/anaconda3/envs/fixed_env/bin/python fetch_index_manual.py
```
- Optional minute data (TopN timing):
```bash
cd /Users/haohao/stockquantbot
/opt/homebrew/anaconda3/envs/fixed_env/bin/python update_manual_minute.py --interval m5 --limit 320
```

## 4) Model Training
- Train LightGBM (rolling, no leakage by date split):
```bash
cd /Users/haohao/stockquantbot
python train_lightgbm.py \
  --years 3 \
  --forward-days 1 \
  --task cls \
  --label-mode excess \
  --rolling \
  --train-months 12 --val-months 4 --step-months 2 \
  --recent-weight-months 6 --recent-weight-mult 1.7 \
  --max-symbols 1870 \
  --out models/lightgbm_fd1_excess_y3_compact.json
```

## 5) Backtest
- Pure model TopK:
```bash
cd /Users/haohao/stockquantbot
python backtest_model.py --config config/default.yaml --months 4 --topk 3 --capital 10000
```
- Rule + model:
```bash
cd /Users/haohao/stockquantbot
python backtest_rule_model.py --config config/default.yaml --months 4 --topk 5 --capital 10000
```

## 6) Known Runtime Notes
- `topk` is ranking-based; no hard minimum score unless explicitly configured in strategy/model filter.
- Backtest buyability check includes limit-up open skip (`entry_open < prev_close * (1 + limit_up_pct)`).
- Real-time source instability (proxy/DNS) can force fallback to `hist`, which affects intraday sensitivity.
- Shrink-volume factor definition:
  - `vol_ratio` (daily) = `today_volume / MA20(volume)`.
  - `volume_change_20d` = `log(today_volume / volume_20_days_ago)`.
  - `volume_shrink_20d = 1` iff `volume_change_20d < log(0.9)` (i.e. current volume is >10% lower than 20 trading days ago).
- Optional layered preselect:
  - Enable with `signals.preselect_layered: true`.
  - Uses `signals.preselect_layer_col` (default `mkt_cap`) + `preselect_layer_bins` + `preselect_layer_weights`.
  - If cap data is missing/invalid, it auto-falls back to plain amount TopN.

## 7) Server Deployment (Ubuntu + Nginx + Systemd)
- Target public host example: `8.129.235.203`
- The API now serves `web/dist` at `/` in production; frontend API calls are same-origin by default.

1. Upload code to server (example path: `/opt/stockquantbot`), then run:
```bash
cd /opt/stockquantbot
sudo APP_DIR=/opt/stockquantbot PUBLIC_HOST=8.129.235.203 bash deploy/deploy_ubuntu.sh
```

2. Open security group / firewall ports:
```bash
22/tcp, 80/tcp
```

3. Verify service:
```bash
curl http://8.129.235.203/api/health
```

4. Common service commands:
```bash
sudo systemctl status stockquantbot
sudo journalctl -u stockquantbot -f
sudo systemctl restart stockquantbot
sudo nginx -t && sudo systemctl reload nginx
```
