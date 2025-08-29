#!/usr/bin/env python3
"""
Visual PyBullet simulation test - No external URDF files needed.
This will show you a 3D simulation with falling objects.
"""

import time

def test_visual_simulation():
    """Test PyBullet with visual objects - no URDF files needed."""
    print("🎬 Starting Visual PyBullet Simulation Test...")
    
    try:
        import pybullet as p
        import pybullet_data
        
        # Connect to GUI
        physics_client = p.connect(p.GUI)
        
        # Add search path for built-in URDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        print("✅ PyBullet GUI opened!")
        print("🎮 Use mouse to rotate view, scroll to zoom")
        
        # Set up physics
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        # Create ground plane (simple box)
        ground_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2, 2, 0.1])
        ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[2, 2, 0.1], rgbaColor=[0.5, 0.5, 0.5, 1])
        ground_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=ground_collision,
            baseVisualShapeIndex=ground_visual,
            basePosition=[0, 0, -0.1]
        )
        print("🌍 Ground plane created")
        
        # Create colorful objects
        objects = []
        
        # Red cube
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
        cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[1, 0, 0, 1])
        cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=[0.2, 0.2, 1.0]
        )
        objects.append(("Red Cube", cube_id))
        
        # Blue sphere
        sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 1])
        sphere_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=[-0.2, 0.2, 1.2]
        )
        objects.append(("Blue Sphere", sphere_id))
        
        # Green cylinder
        cylinder_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.04, height=0.1)
        cylinder_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=0.1, rgbaColor=[0, 1, 0, 1])
        cylinder_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cylinder_collision,
            baseVisualShapeIndex=cylinder_visual,
            basePosition=[0, -0.2, 1.1]
        )
        objects.append(("Green Cylinder", cylinder_id))
        
        # Yellow box (different size)
        box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.06, 0.04])
        box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.06, 0.04], rgbaColor=[1, 1, 0, 1])
        box_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=box_collision,
            baseVisualShapeIndex=box_visual,
            basePosition=[-0.1, -0.1, 1.3]
        )
        objects.append(("Yellow Box", box_id))
        
        print("📦 Created colorful objects:")
        for name, obj_id in objects:
            print(f"   ✅ {name}")
        
        print("\n🎬 Running 10-second physics simulation...")
        print("👀 Watch the objects fall and settle!")
        
        # Run simulation
        for i in range(600):  # 10 seconds at 60 FPS
            p.stepSimulation()
            time.sleep(1./60.)
            
            # Print status every 2 seconds
            if i % 120 == 0 and i > 0:
                print(f"⏱️  {i//60} seconds elapsed...")
                
                # Print object positions
                for name, obj_id in objects:
                    pos, orn = p.getBasePositionAndOrientation(obj_id)
                    print(f"   {name}: height = {pos[2]:.3f}")
        
        print("\n🎉 Simulation completed successfully!")
        print("💡 If you saw colorful objects falling and settling, PyBullet is working perfectly!")
        
        # Keep window open for inspection
        print("\n🔍 Simulation window will stay open for 30 seconds for inspection...")
        print("   Use mouse to rotate, scroll to zoom, explore the scene!")
        
        time.sleep(30)
        
        # Disconnect
        p.disconnect()
        print("✅ PyBullet simulation closed")
        return True
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        try:
            p.disconnect()
        except:
            pass
        return False

def main():
    """Main test function."""
    print("🚀 PyBullet Visual Simulation Test")
    print("=" * 50)
    
    success = test_visual_simulation()
    
    if success:
        print("\n🎉 SUCCESS! Your PyBullet system is working!")
        print("\n🚀 Ready for robot testing! Next steps:")
        print("   1. python examples/basic_demo.py")
        print("   2. python scripts/interactive_chat.py")
    else:
        print("\n❌ Test failed. Check the error messages above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 