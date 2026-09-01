@echo off
cd /d "%~dp0\.."
streamlit run volforge_dashboard.py --server.address 0.0.0.0
