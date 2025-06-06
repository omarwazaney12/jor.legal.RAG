#!/usr/bin/env python3
"""
Conversation Memory Test Script
Test the enhanced conversation memory features of the Legal RAG System
"""

import requests
import json
import time

# Server URL
BASE_URL = "http://localhost:5003"

def send_query(query):
    """Send a query to the legal system"""
    print(f"\n🔍 Sending: {query}")
    print("-" * 50)
    
    response = requests.post(f"{BASE_URL}/query", 
                           json={"query": query},
                           headers={"Content-Type": "application/json"})
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            result = data.get('result', {})
            answer = result.get('answer', '')
            confidence = result.get('confidence', 0)
            query_type = result.get('query_type', '')
            
            print(f"💡 Response (Confidence: {confidence:.2f}, Type: {query_type}):")
            print(answer)
            
            return True
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
            return False
    else:
        print(f"❌ HTTP Error: {response.status_code}")
        return False

def test_conversation_memory():
    """Test conversation memory with follow-up questions"""
    print("🧪 Testing Conversation Memory Features")
    print("=" * 60)
    
    # Test 1: Initial question about company formation
    print("\n📋 TEST 1: Company Formation Process")
    send_query("ما هي شروط تأسيس الشركات في الأردن؟")
    time.sleep(2)
    
    # Test 2: Follow-up question referencing previous response
    print("\n📋 TEST 2: Follow-up about specific requirement")
    send_query("ما هو الحد الأدنى لرأس المال المطلوب؟")
    time.sleep(2)
    
    # Test 3: Reference to previous points
    print("\n📋 TEST 3: Reference to numbered points")
    send_query("أخبرني أكثر عن النقطة الثانية")
    time.sleep(2)
    
    # Test 4: New topic - Consumer Rights
    print("\n📋 TEST 4: New Topic - Consumer Rights")
    send_query("ما هي حقوق المستهلك في الأردن؟")
    time.sleep(2)
    
    # Test 5: Follow-up about consumer rights
    print("\n📋 TEST 5: Follow-up about consumer protection")
    send_query("كيف يمكنني تقديم شكوى ضد متجر؟")
    time.sleep(2)
    
    # Test 6: Reference to previous consumer answer
    print("\n📋 TEST 6: Reference to previous context")
    send_query("ما هي المدة المحددة لهذا الإجراء؟")
    time.sleep(2)
    
    # Test 7: Switch back to company topic
    print("\n📋 TEST 7: Switch back to previous topic")
    send_query("بالعودة لموضوع الشركات، ما هي أنواع الشركات المتاحة؟")
    time.sleep(2)
    
    # Test 8: Follow-up with specific reference
    print("\n📋 TEST 8: Specific follow-up")
    send_query("وضح لي المزيد عن شركة المسؤولية المحدودة")
    time.sleep(2)

def test_english_conversation():
    """Test conversation memory in English"""
    print("\n🇺🇸 Testing English Conversation Memory")
    print("=" * 60)
    
    # English conversation test
    send_query("What are the requirements for trademark registration in Jordan?")
    time.sleep(2)
    
    send_query("Tell me more about the second step")
    time.sleep(2)
    
    send_query("What documents are needed for this procedure?")
    time.sleep(2)

def get_conversation_history():
    """Get and display conversation history"""
    print("\n📜 Conversation History:")
    print("-" * 30)
    
    response = requests.get(f"{BASE_URL}/history")
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            history = data.get('history', [])
            session_id = data.get('session_id', 'Unknown')
            
            print(f"Session ID: {session_id}")
            print(f"Total messages: {len(history)}")
            
            for i, msg in enumerate(history[-6:], 1):  # Show last 6 messages
                msg_type = msg.get('type', 'unknown')
                content = msg.get('content', '')[:100] + '...' if len(msg.get('content', '')) > 100 else msg.get('content', '')
                timestamp = msg.get('timestamp', '')
                
                print(f"{i}. [{msg_type.upper()}] {content}")
                print(f"   Time: {timestamp}")
                print()
        else:
            print(f"Error: {data.get('error', 'Unknown error')}")
    else:
        print(f"HTTP Error: {response.status_code}")

def clear_history():
    """Clear conversation history"""
    print("\n🗑️ Clearing conversation history...")
    
    response = requests.post(f"{BASE_URL}/clear-history")
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print("✅ History cleared successfully")
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
    else:
        print(f"❌ HTTP Error: {response.status_code}")

if __name__ == "__main__":
    print("🚀 Legal RAG Conversation Memory Test")
    print("Make sure the server is running on http://localhost:5003")
    print()
    
    # Test Arabic conversation memory
    test_conversation_memory()
    
    # Display conversation history
    get_conversation_history()
    
    # Test English conversation
    test_english_conversation()
    
    # Display final history
    get_conversation_history()
    
    # Optional: Clear history for next test
    print("\n" + "="*60)
    user_input = input("Do you want to clear the conversation history? (y/n): ")
    if user_input.lower() in ['y', 'yes']:
        clear_history()
    
    print("\n✅ Conversation memory test completed!")
    print("\nKey Features Tested:")
    print("• Follow-up questions with context awareness")
    print("• Reference to previous responses")
    print("• Numbered point references ('النقطة الثانية')")
    print("• Topic switching and returning")
    print("• Context-aware explanations")
    print("• Both Arabic and English conversations")
    print("• Conversation history tracking") 