@echo off
rem SmartScalpel Theatre - double-click to start. Keep this window open during the demo.
cd /d "%~dp0"
py -m pip install -q -r requirements-ml.txt -r requirements-web.txt
py server.py %*
pause
