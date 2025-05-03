@echo off
cd C:\Users\VICTUS\smart_cane
call smart_cane_env\Scripts\activate
start cmd /k "uvicorn main:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 5

