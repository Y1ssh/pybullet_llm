# PyBullet LLM Robotics

A comprehensive robotics simulation system that enables natural language control of robots using PyBullet physics simulation and multiple Large Language Model (LLM) providers.

## 🚀 Features

- **Natural Language Robot Control**: Control robots using everyday language
- **Multi-LLM Support**: Switch between Anthropic Claude, OpenAI GPT, and Google Gemini
- **Physics Simulation**: Realistic robot simulation using PyBullet
- **Computer Vision**: Object detection and manipulation using OpenCV
- **KUKA iiwa Robot**: Complete 7-DOF robot arm implementation
- **Interactive Chat Interface**: Real-time conversation with your robot
- **Safety Features**: Collision detection, workspace limits, emergency stop
- **Provider Comparison**: Compare responses from different LLM providers

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)

### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd pybullet_llm_robotics
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Copy the environment configuration:

```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
GOOGLE_API_KEY=your-google-gemini-key-here

# Default settings
DEFAULT_LLM_PROVIDER=anthropic
```

## 🎮 Quick Start

### Interactive Chat Mode

Start the interactive robot control system:

```bash
cd scripts
python interactive_chat.py
```

Or with specific options:

```bash
python interactive_chat.py --provider openai --no-gui
```

### Basic Commands

Once the system is running, try these commands:

```
🎤 You: scan the environment
🎤 You: pick up the red cube
🎤 You: move to position 0.3, 0.4, 0.5
🎤 You: place the object on the table
🎤 You: switch to OpenAI
🎤 You: compare LLM responses for "pick up the blue sphere"
```

## 📚 Usage Examples

### Basic Robot Control

```python
from src.core.environment import RobotEnvironment
from src.robots.kuka_iiwa import KukaIiwa
from src.llm.llm_controller import LLMRobotController
from src.core.sensors import CameraSystem

# Setup environment
env = RobotEnvironment(gui_mode=True)
env.setup_physics()

# Create robot
robot = KukaIiwa()
robot.load_robot(env.physics_client)

# Setup camera and controller
camera = CameraSystem(env)
controller = LLMRobotController(robot, env, camera)

# Control with natural language
response = controller.chat_with_robot("Move the robot to the red cube")
print(response)
```

### Multi-LLM Provider Usage

```python
# Switch between providers
controller.switch_llm_provider("openai")
controller.switch_llm_provider("google")

# Compare responses
comparison = controller.compare_llm_responses("How should I pick up the fragile glass?")
print(comparison)
```

### Computer Vision Integration

```python
from src.vision.object_detection import ObjectDetector

detector = ObjectDetector()
rgb_image = camera.capture_rgb_image()
objects = detector.detect_objects(rgb_image)

for obj in objects:
    print(f"Found {obj['color']} {obj['shape']} at {obj['center']}")
```

## 🏗️ Architecture

```
pybullet_llm_robotics/
├── src/
│   ├── core/              # Core robotics components
│   │   ├── robot_arm.py   # Base robot arm class
│   │   ├── environment.py # PyBullet environment
│   │   └── sensors.py     # Camera and sensors
│   ├── llm/               # LLM integration
│   │   ├── llm_controller.py  # Main LLM controller
│   │   └── robot_tools.py     # LangChain tools
│   ├── vision/            # Computer vision
│   │   └── object_detection.py
│   ├── robots/            # Robot implementations
│   │   ├── kuka_iiwa.py   # KUKA iiwa robot
│   │   └── llm_provider_manager.py
│   └── utils/             # Utilities
│       └── config.py      # Configuration management
├── scripts/               # Executable scripts
│   └── interactive_chat.py
├── data/                  # Robot models and configs
├── tests/                 # Unit tests
└── examples/              # Demo implementations
```

## 🎯 Supported Commands

### Movement Commands
- "Move the robot to position X, Y, Z"
- "Go to the center of the table"
- "Reset robot to home position"

### Object Manipulation
- "Pick up the [color] [shape]"
- "Grab the object on the left"
- "Place the object at position X, Y, Z"
- "Put it on the table"

### Environment Interaction
- "Scan the environment"
- "What objects can you see?"
- "Find the red cube"
- "Count the objects"

### LLM Provider Management
- "Switch to OpenAI"
- "Use Google Gemini"
- "Compare all providers for this task"

### System Commands
- "help" - Show available commands
- "status" - Display system status
- "providers" - List LLM providers
- "quit" - Exit the system

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_LLM_PROVIDER` | Default LLM provider | `anthropic` |
| `LLM_TEMPERATURE` | LLM response temperature | `0.1` |
| `MAX_TOKENS` | Maximum response tokens | `1000` |
| `CAMERA_WIDTH` | Camera image width | `640` |
| `CAMERA_HEIGHT` | Camera image height | `480` |
| `DEBUG_MODE` | Enable debug logging | `false` |

### Robot Configuration

```python
from src.utils.config import robot_config

robot_config.robot_type = "kuka_iiwa"
robot_config.base_position = [0, 0, 0]
robot_config.home_position = [0, 0, 0, -1.57, 0, 1.57, 0]
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run specific tests:

```bash
python -m pytest tests/test_robot_arm.py -v
python -m pytest tests/test_llm_integration.py -v
```

## 🔍 Troubleshooting

### Common Issues

**1. "No LLM providers available"**
- Check your API keys in `.env`
- Ensure required packages are installed: `pip install langchain-anthropic langchain-openai langchain-google-genai`

**2. "Robot URDF not found"**
- The system will create a simple KUKA URDF automatically
- For custom URDFs, place them in `data/robot_models/kuka_iiwa/`

**3. "Camera not working"**
- Ensure OpenCV is installed: `pip install opencv-python`
- Check if running in headless mode (use `--no-gui` flag)

**4. "PyBullet connection failed"**
- Try running without GUI: `python interactive_chat.py --no-gui`
- Check PyBullet installation: `pip install pybullet`

### Debug Mode

Enable debug logging:

```bash
python interactive_chat.py --log-level DEBUG
```

## 📈 Performance Tips

1. **Use appropriate LLM providers**:
   - Anthropic Claude: Best for complex reasoning
   - OpenAI GPT: Balanced performance
   - Google Gemini: Fast responses

2. **Optimize camera settings**:
   ```python
   env.setup_camera(width=320, height=240)  # Lower resolution for speed
   ```

3. **Reduce simulation complexity**:
   ```python
   env.timestep = 0.01  # Larger timestep for faster simulation
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run tests: `python -m pytest`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyBullet team for the physics simulation engine
- LangChain for LLM integration framework
- OpenAI, Anthropic, and Google for LLM APIs
- OpenCV community for computer vision tools

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `/docs`
- Review example implementations in `/examples`

---

**Happy Robot Programming! 🤖** 