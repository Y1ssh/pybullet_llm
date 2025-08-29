#!/usr/bin/env python3
"""
Test script for Gemini 2.0 Flash API integration.
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

def test_gemini_api_direct():
    """Test Gemini 2.0 Flash API key directly using REST API."""
    print("🧪 Testing Gemini 2.0 Flash API...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("Make sure to set your API key in the .env file")
        return False
    
    # Gemini 2.0 Flash endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    # Test payload for robotics context
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "You are controlling a robot arm. Respond with a brief acknowledgment that you understand robot control tasks."
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topK": 1,
            "topP": 1,
            "maxOutputTokens": 100
        }
    }
    
    try:
        print(f"📡 Sending request to Gemini 2.0 Flash...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                generated_text = result['candidates'][0]['content']['parts'][0]['text']
                print("✅ API Test Successful!")
                print(f"🤖 Gemini 2.0 Flash Response: {generated_text}")
                return True
            else:
                print("❌ Unexpected response format")
                print(f"Response: {json.dumps(result, indent=2)}")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Error details: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout - API might be slow or unavailable")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_langchain_integration():
    """Test Gemini integration through LangChain."""
    print("\n🔗 Testing LangChain Integration...")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ API key not found")
            return False
        
        # Initialize with Gemini 2.0 Flash
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-2.0-flash",
            temperature=0.1,
            max_output_tokens=100
        )
        
        # Test robotics prompt
        response = llm.invoke("You are a robot controller. Say 'Robot control system ready' in a brief response.")
        
        print("✅ LangChain Integration Successful!")
        print(f"🤖 Response: {response.content}")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Try: pip install langchain-google-genai")
        return False
    except Exception as e:
        print(f"❌ LangChain test failed: {e}")
        return False

def test_robot_integration():
    """Test full robot integration with Gemini 2.0 Flash."""
    print("\n🦾 Testing Robot Integration...")
    
    try:
        from src.robots.llm_provider_manager import LLMProviderManager
        
        # Initialize provider manager
        manager = LLMProviderManager()
        
        # Check if Google provider is available
        if "google" not in manager.get_available_providers():
            print("❌ Google provider not available")
            return False
        
        # Test provider capabilities
        capabilities = manager.get_provider_capabilities("google")
        print("📋 Google Provider Capabilities:")
        for key, value in capabilities.items():
            print(f"   {key}: {value}")
        
        # Test response comparison
        test_command = "How would you pick up a red cube safely?"
        responses = manager.compare_providers_response(test_command)
        
        if "google" in responses:
            print("✅ Robot Integration Successful!")
            print(f"🤖 Test Response: {responses['google'][:100]}...")
            return True
        else:
            print("❌ No response from Google provider")
            return False
            
    except Exception as e:
        print(f"❌ Robot integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Gemini 2.0 Flash Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Direct API
    api_test = test_gemini_api_direct()
    
    # Test 2: LangChain integration
    langchain_test = test_langchain_integration()
    
    # Test 3: Robot integration
    robot_test = test_robot_integration()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   Direct API Test: {'✅ PASS' if api_test else '❌ FAIL'}")
    print(f"   LangChain Test:  {'✅ PASS' if langchain_test else '❌ FAIL'}")
    print(f"   Robot Test:      {'✅ PASS' if robot_test else '❌ FAIL'}")
    
    if all([api_test, langchain_test, robot_test]):
        print("\n🎉 All tests passed! Gemini 2.0 Flash is ready for robot control!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check your configuration and API key.")
        return 1

if __name__ == "__main__":
    exit(main()) 