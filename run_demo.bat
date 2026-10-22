@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Setting up the demo for first use. This can take several minutes.
  "C:\Users\asus\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
)
call .venv\Scripts\activate.bat
python -c "import fastapi, tensorflow, ultralytics, cv2" >nul 2>nul
if errorlevel 1 (
  echo Installing the demo packages. This can take several minutes.
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
)
if not exist "frontend\node_modules" (
  echo Installing the website packages. This only happens once.
  pushd frontend
  call npm install
  popd
)
start "AI Project API" /min .venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
pushd frontend
call npm run dev
popd
