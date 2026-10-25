@echo off
rem SmartScalpel LED bridge - run on the laptop with the Arduino plugged in.
rem Usage: led_bridge.bat http://192.168.1.23:5000   (the address shown in the server window)
rem        led_bridge.bat http://192.168.1.23:5000 COM5
cd /d "%~dp0"
py -m pip install -q pyserial
if "%~1"=="" (
  set /p SERVER=Theatre server address, e.g. http://192.168.1.23:5000 : 
) else (
  set SERVER=%~1
)
if "%~2"=="" (
  py led_bridge.py --server %SERVER%
) else (
  py led_bridge.py --server %SERVER% --serial %~2
)
pause
