#!/usr/bin/env python3
"""
Simple test script to verify PyBullet simulation is working.
Run this first to see if you can see the simulation window.
"""

import sys
import time

def test_pybullet_basic():
    """Test basic PyBullet functionality."""
    print("🧪 Testing PyBullet Basic Functionality...")
    
    try:
        import pybullet as p
        print("✅ PyBullet imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyBullet: {e}")
        print("💡 Solution: pip install pybullet")
        return False
    
    try:
        # Connect to GUI - this should open a window
        print("🎬 Opening PyBullet GUI window...")
        physics_client = p.connect(p.GUI)
        
        if physics_client < 0:
            print("❌ Failed to connect to PyBullet GUI")
            return False
        
        print("✅ PyBullet GUI connected successfully!")
        print("👀 You should see a PyBullet window open now")
        
        # Set up basic physics
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        print("🌍 Loading ground plane...")
        plane_id = p.loadURDF("plane.urdf")
        
        # Load a simple cube
        print("📦 Loading a test cube...")
        cube_start_pos = [0, 0, 1]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Create a simple cube
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
        cube_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=cube_start_pos,
            baseOrientation=cube_start_orientation
        )
        
        print("🎯 Test cube created - you should see a red cube falling!")
        
        # Run simulation for a few seconds
        print("⏱️  Running simulation for 5 seconds...")
        for i in range(240):  # 240 steps = ~5 seconds at 1/60 timestep
            p.stepSimulation()
            time.sleep(1./60.)
            
            if i % 60 == 0:  # Print every second
                cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
                print(f"   Cube position: ({cube_pos[0]:.2f}, {cube_pos[1]:.2f}, {cube_pos[2]:.2f})")
        
        print("🎉 Basic simulation test completed!")
        print("💡 If you saw the cube fall and settle, PyBullet is working correctly!")
        
        input("Press Enter to close the simulation...")
        
        # Disconnect
        p.disconnect()
        print("✅ PyBullet disconnected")
        return True
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        try:
            p.disconnect()
        except:
            pass
        return False

def test_robot_system():
    """Test the robot system components."""
    print("\n🤖 Testing Robot System Components...")
    
    # Test imports
    required_modules = [
        ("pybullet", "PyBullet physics engine"),
        ("numpy", "NumPy for numerical computing"),
        ("cv2", "OpenCV for computer vision"), 
        ("langchain", "LangChain for LLM integration"),
        ("dotenv", "Python-dotenv for environment variables")
    ]
    
    all_good = True
    for module, description in required_modules:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} - Missing!")
            all_good = False
    
    if not all_good:
        print("\n💡 To install missing packages:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required modules available!")
    
    # Test project structure
    import os
    required_files = [
        "src/core/robot_arm.py",
        "src/core/environment.py", 
        "src/llm/llm_controller.py",
        "scripts/interactive_chat.py",
        "examples/basic_demo.py"
    ]
    
    print("\n📁 Checking project structure...")
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing!")
            all_good = False
    
    return all_good

def main():
    """Main test function."""
    print("🚀 PyBullet LLM Robotics - System Test")
    print("=" * 50)
    
    # Test 1: Basic PyBullet
    pybullet_ok = test_pybullet_basic()
    
    # Test 2: Robot system components
    components_ok = test_robot_system()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   PyBullet Simulation: {'✅ PASS' if pybullet_ok else '❌ FAIL'}")
    print(f"   Robot Components:    {'✅ PASS' if components_ok else '❌ FAIL'}")
    
    if pybullet_ok and components_ok:
        print("\n🎉 All tests passed! Your system is ready!")
        print("\n🚀 Next steps:")
        print("1. Run: python examples/basic_demo.py")
        print("2. Run: python scripts/interactive_chat.py")
        print("3. Try natural language commands!")
    else:
        print("\n⚠️  Some tests failed. Please:")
        print("1. Install Python if not available")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check the SETUP_GUIDE.md for detailed instructions")
    
    return 0 if (pybullet_ok and components_ok) else 1

if __name__ == "__main__":
    exit(main()) 