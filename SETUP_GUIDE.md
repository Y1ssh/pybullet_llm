# PyBullet LLM Robotics - Complete Setup Guide

## 🎯 Goal: Get PyBullet Simulation Visible and Running

This guide will help you install Python, set up the environment, and see the robot simulation in action.

## 🐍 Step 1: Install Python

### Option A: Microsoft Store (Recommended - Easiest)
1. Open Microsoft Store
2. Search for "Python 3.12"
3. Install "Python 3.12" by Python Software Foundation
4. This automatically adds Python to your PATH

### Option B: Direct Download
1. Go to https://www.python.org/downloads/
2. Download Python 3.12.x (latest version)
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Install with default settings

### Option C: Via Package Manager (if available)
```powershell
# If you have Chocolatey
choco install python

# If you have Scoop
scoop install python
```

## ✅ Step 2: Verify Python Installation

Open a new PowerShell window and test:

```powershell
python --version
# Should show: Python 3.12.x

pip --version
# Should show pip version
```

## 📦 Step 3: Install Dependencies

```powershell
# Navigate to project directory
cd "S:\Dev_testing\LLm to robot simulation\pybullet_llm_robotics"

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

## 🔑 Step 4: Configure API Keys

Create a `.env` file:

```powershell
# Copy the example file
Copy-Item env.example .env

# Edit the .env file with your API keys
notepad .env
```

Add your keys to `.env`:
```
GOOGLE_API_KEY=AIzaSyB8Ig4dWIt_YJ12uwKMYa2fmO82ZZrQ2WA
LLM_MODEL_GOOGLE=gemini-2.0-flash
DEFAULT_LLM_PROVIDER=google
```

## 🚀 Step 5: Test PyBullet Simulation

### Quick Visual Test
```powershell
# Test basic PyBullet (should open GUI window)
python -c "import pybullet as p; p.connect(p.GUI); input('Press Enter to close...')"
```

### Basic Demo (Robot + Objects)
```powershell
python examples/basic_demo.py
```

**Expected Result**: 
- ✅ PyBullet GUI window opens
- ✅ You see a table with colored objects (red cube, blue sphere, etc.)
- ✅ KUKA robot arm visible
- ✅ Robot moves and demonstrates functionality

### Full Interactive System
```powershell
python scripts/interactive_chat.py --provider google
```

**Expected Result**:
- ✅ PyBullet GUI with robot and objects
- ✅ Interactive chat interface
- ✅ You can type commands like "scan the environment"

## 🧪 Step 6: Test Commands

Once the interactive system is running, try these commands:

```
🎤 You: scan the environment
🎤 You: pick up the red cube
🎤 You: move to position 0.3, 0.4, 0.5
🎤 You: place the object on the table
🎤 You: reset robot to home position
🎤 You: status
```

## 🔧 Troubleshooting

### Problem: "python" not recognized
**Solution**: 
- Restart PowerShell after installing Python
- Or use full path: `C:\Users\YourName\AppData\Local\Programs\Python\Python312\python.exe`

### Problem: No GUI window appears
**Solutions**:
1. **Check if running in headless mode**:
   ```powershell
   python scripts/interactive_chat.py --no-gui
   ```

2. **Test PyBullet directly**:
   ```powershell
   python -c "
   import pybullet as p
   physicsClient = p.connect(p.GUI)
   print('PyBullet GUI should be visible now!')
   input('Press Enter to continue...')
   p.disconnect()
   "
   ```

3. **Check display settings** - ensure you're not using remote desktop without proper display forwarding

### Problem: Import errors
**Solution**:
```powershell
# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Problem: API key errors
**Solution**:
```powershell
# Test API key directly
python scripts/test_gemini_api.py
```

## 🎮 What You Should See

### 1. PyBullet GUI Window
- 3D physics simulation environment
- Camera controls (mouse to rotate, scroll to zoom)
- Real-time physics simulation

### 2. Robot Scene
- KUKA iiwa robot arm (orange/gray)
- Table surface (brown)
- Colored objects:
  - Red cube
  - Blue sphere  
  - Green cylinder
  - Yellow box

### 3. Interactive Features
- Robot arm moves smoothly
- Objects respond to physics
- Camera captures the scene
- LLM provides intelligent responses

## 🏆 Success Indicators

✅ **PyBullet GUI opens and stays open**
✅ **Robot arm visible and articulated**
✅ **Colored objects on table**
✅ **Robot responds to movement commands**
✅ **Chat interface accepts natural language**
✅ **Gemini 2.0 Flash provides robot control responses**

## 🚨 Quick Debug Commands

```powershell
# Test individual components
python -c "import pybullet; print('PyBullet OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import numpy; print('NumPy OK')"
python -c "from langchain_google_genai import ChatGoogleGenerativeAI; print('LangChain Google OK')"

# Test environment variables
python -c "import os; print('GOOGLE_API_KEY:', 'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET')"
```

## 🎯 Next Steps After Success

1. **Experiment with commands**: Try different natural language instructions
2. **Switch LLM providers**: Test "switch to OpenAI" or "switch to Anthropic"
3. **Compare responses**: Use "compare LLM responses for [command]"
4. **Modify the scene**: Add your own objects or change robot parameters
5. **Extend functionality**: Add new tools or robot capabilities

## 📞 Need Help?

If you encounter issues:
1. Check this troubleshooting guide
2. Verify all prerequisites are met
3. Test each component individually
4. Check the GitHub repository for updates

Happy Robot Programming! 🤖✨ 