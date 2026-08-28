@echo off
cd /d "%~dp0\.."
python scripts/capture_chain.py --provider yahoo --symbols SPY LULU SNOW NVDA GLD IWM --dte 7 180