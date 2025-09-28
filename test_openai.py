#!/usr/bin/env python3
"""
Test OpenAI API connection
"""

import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

def test_openai_direct():
    """Test OpenAI API directly"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    print("🔍 OpenAI API Test")
    print("=" * 30)
    print(f"API Key Present: {'✅ Yes' if api_key else '❌ No'}")
    if api_key:
        print(f"API Key (first 10 chars): {api_key[:10]}...")
    print(f"Model: {model}")
    
    if not api_key:
        print("❌ No API key found. Please check your .env file.")
        return False
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        print("\n📡 Testing API connection...")
        
        # Make a simple test call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! This is a connection test. Please respond with 'Connection successful'."}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        if response and response.choices:
            result = response.choices[0].message.content
            print(f"✅ Connection successful!")
            print(f"📝 Response: {result}")
            return True
        else:
            print("❌ No response received")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openai_direct()
    print(f"\n🎯 Overall Result: {'SUCCESS' if success else 'FAILED'}")