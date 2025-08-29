@echo off
echo 🚀 PyBullet LLM Robotics - Quick Setup
echo =====================================

echo.
echo 🐍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    echo.
    echo 💡 Please install Python first:
    echo    1. Microsoft Store: Search "Python 3.12"
    echo    2. Or visit: https://www.python.org/downloads/
    echo.
    echo After installing Python, run this script again.
    pause
    exit /b 1
)

echo ✅ Python found!
python --version

echo.
echo 📦 Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo 🔑 Setting up environment file...
if not exist .env (
    copy env.example .env
    echo ✅ Created .env file
    echo ⚠️  Remember to add your API keys to .env file!
) else (
    echo ✅ .env file already exists
)

echo.
echo 🧪 Running basic test...
python test_simulation.py

echo.
echo 🎉 Setup complete!
echo.
echo 🚀 To start the robot system:
echo    python scripts/interactive_chat.py
echo.
echo 📖 For detailed help, see SETUP_GUIDE.md
echo.
pause 