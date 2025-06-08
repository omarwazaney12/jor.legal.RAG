#!/usr/bin/env python3
"""
Advanced Legal RAG Web Demo
Web interface for the Advanced Legal RAG System with enhanced features
"""

import os
import uuid
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, session
from flask_cors import CORS

# Import the advanced RAG system
from advanced_rag_system import AdvancedLegalRAGSystem, QueryResult

app = Flask(__name__)
app.secret_key = 'advanced_legal_rag_secret_2024'
CORS(app)

# Global system instance
legal_system: AdvancedLegalRAGSystem = None
conversation_store = {}

def is_inappropriate_content(query: str) -> bool:
    """Check if query contains inappropriate content"""
    inappropriate_patterns = [
        # Profanity in English
        'fuck', 'shit', 'damn', 'bitch', 'ass', 'bastard', 
        'hell', 'piss', 'crap', 'motherfucker', 'asshole',
        # Profanity in Arabic
        'كس', 'زب', 'خرا', 'نيك', 'عرص', 'قحبة', 'زاني',
        # Other inappropriate
        'idiot', 'stupid', 'dumb', 'moron', 'retard'
    ]
    
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in inappropriate_patterns)

def is_test_or_greeting(query: str) -> bool:
    """Check if query is just a test or greeting"""
    test_patterns = [
        # English tests
        'test', 'testing', 'hello', 'hi', 'hey', 'sup', 'yo',
        'how are you', 'what\'s up', 'good morning', 'good evening',
        # Arabic greetings/tests
        'مرحبا', 'أهلا', 'السلام عليكم', 'تست', 'تجربة', 'مساء الخير', 'صباح الخير',
        'كيف حالك', 'شلونك', 'وين', 'شو', 'ايش', 'كيفك'
    ]
    
    query_lower = query.lower().strip()
    
    # Check exact matches for short queries
    if len(query_lower.split()) <= 3:
        return any(pattern in query_lower for pattern in test_patterns)
    
    return False

def is_legal_query(query: str) -> bool:
    """Check if query is related to legal matters"""
    legal_keywords_arabic = [
        'قانون', 'نظام', 'تعليمات', 'شركة', 'تأسيس', 'تسجيل', 'محكمة', 'عقد',
        'حقوق', 'واجبات', 'مسؤولية', 'ضمان', 'تأمين', 'استثمار', 'تجارة',
        'علامة تجارية', 'براءة اختراع', 'ملكية فكرية', 'عمل', 'موظف', 'راتب',
        'إجازة', 'استقالة', 'فصل', 'تعويض', 'غرامة', 'مخالفة', 'جريمة',
        'مدني', 'جنائي', 'تجاري', 'إداري', 'دستوري', 'وزارة', 'حكومة',
        'رخصة', 'تصريح', 'موافقة', 'اعتماد', 'شهادة', 'وثيقة', 'مستند'
    ]
    
    legal_keywords_english = [
        'law', 'legal', 'company', 'business', 'contract', 'agreement',
        'rights', 'obligations', 'liability', 'insurance', 'investment',
        'trademark', 'patent', 'copyright', 'employment', 'labor',
        'salary', 'wage', 'termination', 'resignation', 'compensation',
        'fine', 'penalty', 'crime', 'civil', 'criminal', 'commercial',
        'administrative', 'constitutional', 'ministry', 'government',
        'license', 'permit', 'approval', 'certification', 'document',
        'registration', 'incorporation', 'establishment', 'court',
        'tribunal', 'jurisdiction', 'regulation', 'statute', 'legislation'
    ]
    
    query_lower = query.lower()
    
    # Check for Arabic legal terms
    arabic_matches = sum(1 for keyword in legal_keywords_arabic if keyword in query_lower)
    english_matches = sum(1 for keyword in legal_keywords_english if keyword in query_lower)
    
    # If query has legal keywords, it's likely legal
    if arabic_matches >= 1 or english_matches >= 1:
        return True
    
    # Check for question patterns that might be legal
    legal_question_patterns = [
        'كيف أ', 'كيفية', 'ما هو', 'ما هي', 'متى', 'أين', 'لماذا',
        'how to', 'how do', 'what is', 'what are', 'when', 'where', 'why',
        'can i', 'should i', 'do i need', 'is it legal', 'is it allowed'
    ]
    
    has_question_pattern = any(pattern in query_lower for pattern in legal_question_patterns)
    
    # If it has question patterns and is reasonably long, might be legal
    if has_question_pattern and len(query.split()) >= 3:
        return True
    
    return False

def detect_language(text: str) -> str:
    """Detect if the text is primarily in Arabic or English"""
    import re
    
    # Count Arabic characters
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    
    # Count English/Latin characters
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # Total meaningful characters
    total_meaningful = arabic_chars + english_chars
    
    if total_meaningful == 0:
        return 'arabic'  # Default to Arabic
    
    # If more than 60% Arabic characters, consider it Arabic
    if arabic_chars / total_meaningful > 0.6:
        return 'arabic'
    elif english_chars / total_meaningful > 0.6:
        return 'english'
    else:
        # Mixed or unclear, use some common keywords
        english_keywords = ['what', 'how', 'when', 'where', 'why', 'company', 'law', 'procedure', 'requirements']
        arabic_keywords = ['ما', 'كيف', 'متى', 'أين', 'لماذا', 'شركة', 'قانون', 'إجراءات', 'شروط']
        
        text_lower = text.lower()
        english_found = sum(1 for keyword in english_keywords if keyword in text_lower)
        arabic_found = sum(1 for keyword in arabic_keywords if keyword in text_lower)
        
        return 'english' if english_found > arabic_found else 'arabic'

def clean_response_formatting(response: str) -> str:
    """Clean and standardize response formatting (same as RAG system)"""
    import re
    
    # Remove all markdown formatting
    cleaned = response
    
    # Remove bold formatting (**text**)
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
    
    # Remove italic formatting (*text*)
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
    
    # Remove markdown headers (## text)
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)
    
    # Clean up excessive whitespace first
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    # Process each line individually for precise control
    lines = cleaned.split('\n')
    formatted_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            formatted_lines.append('')
            continue
        
        # Check what type of line this is
        is_numbered = re.match(r'^\d+\.\s+', line)
        is_bullet_already = re.match(r'^•\s+', line)
        is_dash_bullet = re.match(r'^-\s+', line)
        is_reference = 'المرجع' in line or 'اقتباس' in line or 'Legal Reference' in line or 'Quote:' in line
        is_disclaimer = line.startswith('تنبيه') or line.startswith('Important')
        
        # Handle different line types
        if is_numbered:
            # Ensure numbered items have colon at end if they're titles
            if not line.endswith(':') and not is_reference:
                line = re.sub(r'^(\d+)\.\s+([^:\n]+)(?::?)$', r'\1. \2:', line)
            # Add spacing before numbered items (except first one)
            if formatted_lines and formatted_lines[-1] != '':
                formatted_lines.append('')
                formatted_lines.append('')
            formatted_lines.append(line)
            # Add spacing after numbered titles 
            formatted_lines.append('')
            
        elif is_dash_bullet:
            # Convert dash to bullet
            line = re.sub(r'^-\s*', '• ', line)
            formatted_lines.append(line)
            
        elif is_bullet_already:
            # Already has bullet, just keep it
            formatted_lines.append(line)
            
        elif is_reference or is_disclaimer:
            # References and disclaimers don't get bullets
            if formatted_lines and formatted_lines[-1] != '':
                formatted_lines.append('')
            formatted_lines.append(line)
            
        else:
            # Regular content lines (headers, descriptions, etc.) - no bullets
            formatted_lines.append(line)
    
    # Join lines and final cleanup
    cleaned = '\n'.join(formatted_lines)
    
    # Remove trailing spaces
    cleaned = '\n'.join(line.rstrip() for line in cleaned.split('\n'))
    
    # Final cleanup of excessive spacing
    cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)
    
    return cleaned.strip()

def get_appropriate_response(query: str, issue_type: str) -> str:
    """Generate appropriate response for non-legal queries"""
    user_language = detect_language(query)
    
    if issue_type == 'inappropriate':
        if user_language == 'english':
            response = """🚫 Professional Communication Notice

I am a professional legal consultation system designed to assist with Jordanian legal matters. Please maintain respectful communication.

I can help you with:
• Company registration and business laws
• Consumer protection rights
• Employment and labor laws
• Investment regulations
• Trademark and intellectual property
• Legal procedures and requirements

Important Disclaimer: This system provides general information only and is not a substitute for professional legal advice. Please consult a qualified attorney for personal legal matters.

Please ask your legal question in a respectful manner."""
        else:
            response = """🚫 تنبيه التواصل المهني

أنا نظام استشارات قانونية مهني مخصص لمساعدتك في المسائل القانونية الأردنية. يرجى الحفاظ على التواصل المحترم.

يمكنني مساعدتك في:
• تسجيل الشركات وقوانين الأعمال
• حقوق المستهلك
• قوانين العمل والعمال
• أنظمة الاستثمار
• العلامات التجارية والملكية الفكرية
• الإجراءات والمتطلبات القانونية

تنبيه مهم: هذا النظام يوفر معلومات عامة فقط وليس بديلاً عن الاستشارة القانونية المهنية. يرجى استشارة محام مؤهل للمسائل القانونية الشخصية.

يرجى طرح استفسارك القانوني بطريقة محترمة."""
    
    elif issue_type == 'greeting':
        if user_language == 'english':
            response = """👋 Welcome to the Jordan Legal Assistant

Hello! I am a specialized legal consultation system for Jordanian laws with extensive experience in all legal fields.

My specializations include:
• Consumer Protection Law
• Corporate and Investment Laws
• Commercial and Registration Laws
• Labor and Social Security Laws
• Civil and Criminal Laws
• Administrative and Constitutional Laws

How to use this system:
• Ask specific legal questions about Jordanian law
• Request information about legal procedures
• Inquire about your rights and obligations
• Get guidance on business and investment matters

Important Disclaimer: This system provides general information only and is not a substitute for professional legal advice. Please consult a qualified attorney for personal legal matters.

Example questions you can ask:
• How do I register a company in Jordan?
• What are my consumer rights?
• What are the labor law requirements?

Please ask your legal question and I will provide appropriate consultation according to applicable Jordanian legislation."""
        else:
            response = """👋 مرحباً بك في المساعد القانوني الأردني

أهلاً وسهلاً! أنا نظام استشارات قانونية متخصص في القوانين الأردنية مع خبرة واسعة في جميع المجالات القانونية.

تخصصاتي تشمل:
• قانون حماية المستهلك
• قوانين الشركات والاستثمار
• القوانين التجارية والتسجيل
• قوانين العمل والضمان الاجتماعي
• القوانين المدنية والجنائية
• القوانين الإدارية والدستورية

كيفية استخدام هذا النظام:
• اطرح أسئلة قانونية محددة حول القانون الأردني
• اطلب معلومات عن الإجراءات القانونية
• استفسر عن حقوقك وواجباتك
• احصل على إرشادات في مسائل الأعمال والاستثمار

تنبيه مهم: هذا النظام يوفر معلومات عامة فقط وليس بديلاً عن الاستشارة القانونية المهنية. يرجى استشارة محام مؤهل للمسائل القانونية الشخصية.

أمثلة على الأسئلة التي يمكنك طرحها:
• كيف أسجل شركة في الأردن؟
• ما هي حقوقي كمستهلك؟
• ما هي متطلبات قانون العمل؟

يرجى طرح استفسارك القانوني وسأقدم لك الاستشارة المناسبة وفقاً للتشريعات الأردنية المعمول بها."""
    
    elif issue_type == 'non_legal':
        if user_language == 'english':
            response = """ℹ️ Legal Consultation System

I am specifically designed to provide legal consultation on Jordanian laws and regulations. Your question doesn't appear to be related to legal matters.

I can help you with legal questions such as:
• Company formation and business registration
• Consumer rights and protection
• Employment and labor law issues
• Investment and commercial regulations
• Intellectual property and trademarks
• Legal procedures and requirements
• Government licensing and permits

Important Disclaimer: This system provides general information only and is not a substitute for professional legal advice. Please consult a qualified attorney for personal legal matters.

If you have a legal question about Jordanian law, please feel free to ask!"""
        else:
            response = """ℹ️ نظام الاستشارات القانونية

أنا مصمم خصيصاً لتقديم الاستشارات القانونية في القوانين والأنظمة الأردنية. يبدو أن سؤالك غير متعلق بالمسائل القانونية.

يمكنني مساعدتك في الأسئلة القانونية مثل:
• تأسيس الشركات وتسجيل الأعمال
• حقوق المستهلك والحماية
• قضايا قانون العمل والعمال
• أنظمة الاستثمار والتجارة
• الملكية الفكرية والعلامات التجارية
• الإجراءات والمتطلبات القانونية
• التراخيص والتصاريح الحكومية

تنبيه مهم: هذا النظام يوفر معلومات عامة فقط وليس بديلاً عن الاستشارة القانونية المهنية. يرجى استشارة محام مؤهل للمسائل القانونية الشخصية.

إذا كان لديك سؤال قانوني حول القانون الأردني، يرجى طرحه بكل حرية!"""
    
    else:
        response = "يرجى طرح سؤال قانوني مناسب."
    
    # Apply the same formatting cleanup as RAG responses
    return clean_response_formatting(response)

def validate_and_filter_query(query: str, conversation_context: str = "") -> dict:
    """Validate and filter query before processing"""
    # Basic validation
    if not query or len(query.strip()) < 2:
        print(f"📝 Validation: Empty or too short query")
        return {
            'valid': False,
            'response': 'يرجى إدخال استفسار صحيح.',
            'should_process': False
        }
    
    # Check for inappropriate content
    if is_inappropriate_content(query):
        print(f"🚫 Validation: Inappropriate content detected")
        return {
            'valid': True,
            'response': get_appropriate_response(query, 'inappropriate'),
            'should_process': False
        }
    
    # Check for test/greeting
    if is_test_or_greeting(query):
        print(f"👋 Validation: Greeting/test detected")
        return {
            'valid': True,
            'response': get_appropriate_response(query, 'greeting'),
            'should_process': False
        }
    
    # Check if it's a follow-up question (should bypass legal query check)
    if is_follow_up_question(query, conversation_context):
        print(f"🔄 Validation: Follow-up question detected - bypassing legal query check")
        return {
            'valid': True,
            'response': None,
            'should_process': True
        }
    
    # Check if it's a legal query
    if not is_legal_query(query):
        print(f"ℹ️ Validation: Non-legal query detected")
        return {
            'valid': True,
            'response': get_appropriate_response(query, 'non_legal'),
            'should_process': False
        }
    
    # Query is valid and should be processed
    print(f"✅ Validation: Legal query approved for processing")
    return {
        'valid': True,
        'response': None,
        'should_process': True
    }

def initialize_system():
    """Initialize the advanced legal system"""
    global legal_system
    
    print("🚀 Initializing Advanced Legal RAG System...")
    try:
        legal_system = AdvancedLegalRAGSystem()
        
        # Load all documents
        num_docs = legal_system.load_documents()
        
        if num_docs > 0:
            print(f"✅ System ready with {num_docs} documents!")
            return True
        else:
            print("⚠️ No documents loaded - system will start in limited mode")
            print("🔧 Upload ChromaDB data via shell to enable full functionality")
            # Still initialize the system for web interface
            return True
    except Exception as e:
        print(f"⚠️ System initialization warning: {e}")
        print("🔧 System will start in limited mode - upload ChromaDB data to enable full functionality")
        # Initialize a minimal system
        legal_system = None
        return True

def get_or_create_session():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        conversation_store[session['session_id']] = []
    return session['session_id']

def add_to_conversation(session_id: str, message_type: str, content: str, metadata: dict = None):
    """Add message to conversation history"""
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    
    conversation_store[session_id].append({
        'type': message_type,
        'content': content,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 20 messages
    if len(conversation_store[session_id]) > 20:
        conversation_store[session_id] = conversation_store[session_id][-20:]

def get_conversation_context(session_id: str, limit: int = 3) -> str:
    """Get recent conversation history for context-aware responses"""
    if session_id not in conversation_store:
        return ""
    
    messages = conversation_store[session_id]
    recent_messages = messages[-(limit*2):]  # Get last few Q&A pairs
    
    context_parts = []
    for msg in recent_messages:
        if msg['type'] == 'user':
            context_parts.append(f"Previous User Question: {msg['content']}")
        elif msg['type'] == 'assistant':
            # Use first 800 chars of assistant response for context (more detailed)
            response_snippet = msg['content'][:800]
            context_parts.append(f"Previous Assistant Response: {response_snippet}")
    
    return " | ".join(context_parts)

def is_follow_up_question(query: str, conversation_context: str) -> bool:
    """Detect if this is a follow-up question referencing previous conversation"""
    follow_up_keywords = [
        # Direct references
        'هذا', 'ذلك', 'هذه', 'تلك', 'المذكور', 'السابق', 'المشار إليه',
        # Request for clarification/details
        'أخبرني أكثر', 'وضح', 'اشرح', 'تفصيل', 'تفاصيل', 'بالتفصيل',
        # Numbered references (Arabic)
        'النقطة الأولى', 'النقطة الثانية', 'النقطة الثالثة', 'البند الأول', 'البند الثاني',
        'الخطوة الأولى', 'الخطوة الثانية', 'الخطوة الثالثة',
        # Ordinal references (Arabic)
        'الأول', 'الثاني', 'الثالث', 'الرابع', 'الخامس',
        # Sequential questions
        'وماذا عن', 'وما هو', 'وما هي', 'وكيف', 'ومتى', 'وأين',
        # Continuation words
        'أيضاً', 'كذلك', 'بالإضافة', 'علاوة على ذلك',
        # English follow-up phrases
        'tell me about', 'tell me more', 'explain', 'clarify', 'elaborate',
        'what about', 'regarding', 'concerning', 'about this', 'about that',
        # English numbered references
        'point 1', 'point 2', 'point 3', 'point 4', 'point 5',
        'first point', 'second point', 'third point', 'fourth point', 'fifth point',
        'step 1', 'step 2', 'step 3', 'step 4', 'step 5',
        'first step', 'second step', 'third step', 'fourth step', 'fifth step',
        '1st point', '2nd point', '3rd point', '4th point', '5th point',
        'the first', 'the second', 'the third', 'the fourth', 'the fifth'
    ]
    
    query_lower = query.lower()
    
    # Check for direct keyword matches
    has_follow_up_keywords = any(keyword in query_lower for keyword in follow_up_keywords)
    
    # Check for numbered patterns (1, 2, 3, etc.)
    import re
    numbered_patterns = [
        r'\b(point|step|item|section)\s*[1-5]\b',
        r'\b[1-5](st|nd|rd|th)\s*(point|step|item|section)\b',
        r'\bthe\s+[1-5](st|nd|rd|th)\b',
        r'\b(first|second|third|fourth|fifth)\s*(point|step|item|section)?\b'
    ]
    
    has_numbered_reference = any(re.search(pattern, query_lower) for pattern in numbered_patterns)
    
    # Check if conversation context exists
    has_conversation_context = len(conversation_context) > 0
    
    # Consider it a follow-up if:
    # 1. Has explicit follow-up keywords
    # 2. Has numbered references 
    # 3. Is a short question with conversation context (likely a follow-up)
    is_short_with_context = has_conversation_context and len(query.split()) < 8
    
    return has_follow_up_keywords or has_numbered_reference or is_short_with_context

# Enhanced HTML template with advanced features
ADVANCED_HTML_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>المساعد القانوني الأردني | Jordan Legal Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        html {
            height: 100%;
            overflow: hidden !important;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', 'Noto Sans Arabic', sans-serif;
            background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
            color: #ffffff;
            font-size: 16px;
            height: 100vh;
            width: 100vw;
            overflow: hidden !important;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
        }
        
        .app-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            width: 100vw;
            max-width: none;
            margin: 0;
            padding: 0;
            overflow: hidden !important;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
        }
        
        /* Compact Header */
        .chat-header {
            background: linear-gradient(135deg, #1e40af 0%, #3730a3 50%, #5b21b6 100%);
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            flex-shrink: 0;
            position: relative;
            z-index: 10;
            width: 100%;
        }
        
        .chat-header {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            text-align: center;
        }
        
        .header-left {
            color: white;
            font-size: 1.8em;
            font-weight: 700;
            margin-right: 15px;
        }
        
        .header-center {
            font-size: 2em;
            margin: 0 15px;
        }
        
        .header-right {
            color: white;
            font-size: 1.8em;
            font-weight: 700;
            margin-left: 15px;
        }
        
        /* Main content area - fixed height */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0f172a;
            height: calc(100vh - 90px); /* Fixed height minus header */
            width: 100%;
            overflow: hidden !important;
            position: relative;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 15px 20px;
            background: linear-gradient(to bottom, #0f172a 0%, #1e293b 100%);
            height: calc(100% - 70px); /* Fixed height minus input area */
            width: 100%;
            position: relative;
        }
        
        /* Hide scrollbar but keep functionality */
        .chat-messages::-webkit-scrollbar {
            width: 0px;
            background: transparent;
        }
        
        .chat-messages {
            scrollbar-width: none; /* Firefox */
            -ms-overflow-style: none; /* IE/Edge */
        }
        
        .message {
            margin-bottom: 15px;
            animation: slideInUp 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        
        @keyframes slideInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            text-align: right;
        }
        
        .message.assistant {
            text-align: left;
        }
        
        .message-bubble {
            display: inline-block;
            max-width: 80%;
            padding: 18px 24px;
            border-radius: 20px;
            line-height: 1.6;
            font-size: 0.95em;
            position: relative;
            box-shadow: 0 3px 12px rgba(0,0,0,0.2);
            word-wrap: break-word;
            white-space: pre-line;
        }
        
        .message.user .message-bubble {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            border-bottom-right-radius: 6px;
            margin-right: 8px;
        }
        
        .message.assistant .message-bubble {
            background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
            color: #f1f5f9;
            border-bottom-left-radius: 6px;
            border: 1px solid #6b7280;
            margin-left: 8px;
            direction: rtl;
            text-align: right;
        }
        
        /* Enhanced formatting for legal responses */
        .message.assistant .message-bubble h1,
        .message.assistant .message-bubble h2,
        .message.assistant .message-bubble h3 {
            margin: 15px 0 10px 0;
            font-weight: 600;
            color: #e2e8f0;
            direction: rtl;
            text-align: right;
        }
        
        .message.assistant .message-bubble p {
            margin: 8px 0;
            direction: rtl;
            text-align: right;
            line-height: 1.7;
        }
        
        .message.assistant .message-bubble ul,
        .message.assistant .message-bubble ol {
            margin: 12px 0;
            padding-right: 20px;
            direction: rtl;
            text-align: right;
        }
        
        .message.assistant .message-bubble li {
            margin: 8px 0;
            line-height: 1.6;
            direction: rtl;
            text-align: right;
        }
        
        /* Numbered list styling */
        .message.assistant .message-bubble div[style*="margin"] {
            margin: 15px 0 !important;
        }
        
        /* English content styling */
        .message.assistant .message-bubble.english {
            direction: ltr;
            text-align: left;
        }
        
        .message.assistant .message-bubble.english h1,
        .message.assistant .message-bubble.english h2,
        .message.assistant .message-bubble.english h3,
        .message.assistant .message-bubble.english p,
        .message.assistant .message-bubble.english li {
            direction: ltr;
            text-align: left;
        }
        
        /* Timestamp */
        .message-time {
            font-size: 0.75em;
            opacity: 0.6; 
            margin-top: 6px;
        }
        
        .message.user .message-time {
            text-align: right;
        }
        
        .message.assistant .message-time {
            text-align: left;
        }
        
        /* Welcome section - more compact */
        .welcome-section {
            text-align: center;
            padding: 20px 20px 15px;
            color: #cbd5e1;
            width: 100%;
        }
        
        .welcome-section h3 {
            color: #f1f5f9;
            margin-bottom: 8px;
            font-size: 1.3em;
            font-weight: 600;
        }
        
        .welcome-section .subtitle {
            color: #a78bfa;
            font-size: 0.9em;
            margin-bottom: 12px;
            font-weight: 500;
        }
        
        .welcome-section p {
            font-size: 0.95em;
            line-height: 1.4;
            margin-bottom: 15px;
            color: #94a3b8;
            max-width: 100%;
            margin-left: auto;
            margin-right: auto;
        }
        
        /* Sample questions - better spacing */
        .sample-questions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
            justify-content: center;
            width: 100%;
            padding: 0 10px;
        }
        
        .sample-question {
            background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
            border-radius: 14px;
            padding: 8px 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid #6b7280;
            font-size: 0.85em;
            text-align: center;
            color: #e2e8f0;
            white-space: nowrap;
        }
        
        .sample-question:hover {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            transform: translateY(-1px);
            box-shadow: 0 3px 8px rgba(59, 130, 246, 0.3);
        }
        
        /* Fixed input area */
        .chat-input {
            background: #1e2329;
            border-top: 2px solid #2d3748;
            padding: 12px 20px 15px;
            position: sticky;
            bottom: 0;
            z-index: 100;
            width: 100%;
        }
        
        .input-container {
            display: flex;
            align-items: center;
            background: #2d3748;
            border: 2px solid #4a90e2;
            border-radius: 25px;
            margin: 0;
            padding: 12px 16px;
            box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            max-width: none;
            min-height: 60px;
        }
        
        .input-container:focus-within {
            border-color: #60a5fa;
            box-shadow: 0 4px 20px rgba(96, 165, 250, 0.4);
            background: #374151;
        }
        
        .message-input {
            flex: 1;
            background: transparent;
            border: none;
            color: #ffffff;
            font-size: 1em;
            outline: none;
            padding: 8px 12px;
            min-height: 24px;
            max-height: 120px;
            font-family: inherit;
            direction: rtl;
            text-align: right;
            resize: none;
            overflow-y: auto;
            line-height: 1.4;
        }
        
        .message-input::placeholder {
            color: #9ca3af;
            opacity: 0.8;
        }
        
        .send-button {
            background: linear-gradient(135deg, #4a90e2, #357abd);
            border: none;
            color: white;
            width: 46px;
            height: 46px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: all 0.3s ease;
            margin-left: 8px;
            flex-shrink: 0;
            align-self: flex-end;
        }
        
        .send-button:hover {
            background: linear-gradient(135deg, #357abd, #2563eb);
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(74, 144, 226, 0.4);
        }
        
        .send-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        /* Loading states */
        .typing-indicator {
            text-align: left;
            margin-bottom: 15px;
        }
        
        .typing-bubble {
            display: inline-block;
            background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
            padding: 14px 20px;
            border-radius: 18px;
            border-bottom-left-radius: 6px;
            border: 1px solid #6b7280;
            margin-left: 8px;
            color: #f1f5f9;
            font-size: 0.9em;
        }
        
        .typing-dots {
            display: inline-flex;
            gap: 5px;
            align-items: center;
            margin-top: 5px;
        }
        
        .typing-dots span {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #3b82f6;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1.2); opacity: 1; }
        }
        
        /* Responsive design - Full width on all devices */
        @media (max-width: 1200px) {
            .chat-header {
                padding: 15px 20px;
            }
            
            .chat-messages {
                padding: 15px 20px;
            }
            
            .chat-input {
                padding: 12px 20px;
            }
            
            .sample-questions {
                padding: 0 15px;
            }
            
            .welcome-section {
                padding: 20px 20px 15px;
            }
        }
        
        @media (max-width: 992px) {
            .chat-header h1 {
                font-size: 1.5em;
            }
            
            .chat-header p {
                font-size: 0.85em;
            }
            
            .welcome-section h3 {
                font-size: 1.2em;
            }
            
            .welcome-section .subtitle {
                font-size: 0.85em;
            }
            
            .welcome-section p {
                font-size: 0.9em;
            }
            
            .sample-question {
                font-size: 0.8em;
                padding: 7px 14px;
            }
        }
        
        @media (max-width: 768px) {
            .chat-header {
                padding: 12px 15px;
            }
            
            .chat-header h1 {
                font-size: 1.4em;
            }
            
            .chat-header p {
                font-size: 0.8em;
            }
            
            .chat-messages {
                padding: 12px 15px;
            }
            
            .chat-input {
                padding: 10px 15px;
            }
            
            .message-bubble {
                max-width: 85%;
                font-size: 0.9em;
                padding: 16px 20px;
                line-height: 1.7;
            }
            
            .message.assistant .message-bubble h1,
            .message.assistant .message-bubble h2,
            .message.assistant .message-bubble h3 {
                margin: 12px 0 8px 0;
            }
            
            .message.assistant .message-bubble p {
                margin: 6px 0;
            }
            
            .message.assistant .message-bubble li {
                margin: 6px 0;
            }
            
            .sample-questions {
                gap: 6px;
                margin: 0 10px 15px;
                justify-content: center;
            }
            
            .sample-question {
                font-size: 0.75em;
                padding: 6px 12px;
                flex: 0 1 auto;
                min-width: 120px;
                max-width: 200px;
            }
            
            .welcome-section {
                padding: 12px 15px 10px;
            }
            
            .welcome-section h3 {
                font-size: 1.1em;
                margin-bottom: 6px;
            }
            
            .welcome-section .subtitle {
                font-size: 0.8em;
                margin-bottom: 10px;
            }
            
            .welcome-section p {
                font-size: 0.85em;
                line-height: 1.3;
                margin-bottom: 12px;
            }
            
            .input-container {
                padding: 10px 14px;
                min-height: 55px;
            }
            
            .message-input {
                font-size: 0.9em;
                min-height: 20px;
                padding: 6px 10px;
            }
            
            .send-button {
                width: 40px;
                height: 40px;
                font-size: 15px;
            }
        }
        
        @media (max-width: 480px) {
            .chat-header {
                padding: 10px 12px;
            }
            
            .chat-header h1 {
                font-size: 1.3em;
            }
            
            .chat-header p {
                font-size: 0.75em;
            }
            
            .chat-messages {
                padding: 10px 12px;
            }
            
            .chat-input {
                padding: 8px 12px;
            }
            
            .welcome-section {
                padding: 10px 12px 8px;
            }
            
            .welcome-section h3 {
                font-size: 1em;
                margin-bottom: 5px;
            }
            
            .welcome-section .subtitle {
                font-size: 0.75em;
                margin-bottom: 8px;
            }
            
            .welcome-section p {
                font-size: 0.8em;
                line-height: 1.25;
                margin-bottom: 10px;
            }
            
            .sample-questions {
                gap: 5px;
                margin: 0 5px 12px;
                flex-direction: column;
                align-items: center;
            }
            
            .sample-question {
                width: 100%;
                max-width: 280px;
                font-size: 0.7em;
                padding: 6px 10px;
                text-align: center;
            }
            
            .input-container {
                padding: 8px 12px;
                min-height: 50px;
            }
            
            .message-input {
                font-size: 0.85em;
                min-height: 18px;
                padding: 5px 8px;
            }
            
            .send-button {
                width: 38px;
                height: 38px;
                font-size: 14px;
            }
            
            .message-bubble {
                max-width: 90%;
                font-size: 0.85em;
                padding: 14px 18px;
                line-height: 1.6;
            }
            
            .message.assistant .message-bubble h1,
            .message.assistant .message-bubble h2,
            .message.assistant .message-bubble h3 {
                margin: 10px 0 6px 0;
                font-size: 1.1em;
            }
            
            .message.assistant .message-bubble p {
                margin: 5px 0;
            }
            
            .message.assistant .message-bubble li {
                margin: 5px 0;
            }
        }
        
        .welcome-section .notice {
            background: rgba(255, 193, 7, 0.1);
            border: 1px solid rgba(255, 193, 7, 0.3);
            border-radius: 8px;
            padding: 12px 16px;
            margin: 15px auto;
            font-size: 0.85em;
            line-height: 1.4;
            color: #fbbf24;
            max-width: 100%;
            text-align: center;
        }
        
        .arabic-text {
            direction: rtl;
            text-align: right;
            display: block;
            margin-bottom: 5px;
        }
        
        .english-text {
            direction: ltr;
            text-align: left;
            display: block;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Compact Header -->
        <div class="chat-header">
            <div style="text-align: center;">
                <h1 style="margin: 0; font-size: 1.8em; font-weight: 700; color: white;">
                    المساعد القانوني الأردني | Jordan Legal Assistant
                </h1>
                <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #a78bfa; font-weight: 500;">
                    نسخة تجريبية | Beta Version
                </p>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <div class="chat-messages" id="messages">
                <div class="welcome-section">
                    <p class="notice">
                        <span class="arabic-text">⚠️ تنبيه مهم: هذا النظام للمعلومات العامة فقط وليس بديلاً عن الاستشارة القانونية المهنية.</span><br/>
                        <span class="english-text">⚠️ Important Notice: This system provides general information only and is not a substitute for professional legal advice.</span>
                    </p>
                </div>
                
                <div class="sample-questions">
                    <button class="sample-question" onclick="askQuestion('ما هي شروط تأسيس الشركات؟')">شروط تأسيس الشركات 🏢</button>
                    <button class="sample-question" onclick="askQuestion('ما هي حقوق المستهلك في الأردن؟')">حقوق المستهلك 📍</button>
                    <button class="sample-question" onclick="askQuestion('كيف أسجل علامة تجارية؟')">تسجيل العلامات التجارية ®</button>
                    <button class="sample-question" onclick="askQuestion('ما هي قوانين الاستثمار الأجنبي؟')">قوانين الاستثمار 💰</button>
                    <button class="sample-question" onclick="askQuestion('إجراءات التسجيل التجاري')">التسجيل التجاري 📋</button>
                    <button class="sample-question" onclick="askQuestion('قوانين العمل والعمال')">قوانين العمل 👔</button>
                </div>
                
                <div style="text-align: center; margin-top: 10px;">
                    <button 
                        onclick="clearChatHistory()" 
                        style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171; padding: 6px 12px; border-radius: 8px; font-size: 0.8em; cursor: pointer;"
                        title="Clear conversation history | مسح تاريخ المحادثة"
                    >
                        🗑️ بدء محادثة جديدة | New Chat
                    </button>
                </div>
            </div>
            
            <div class="chat-input">
                <!-- Input Area -->
                <div class="input-container">
                    <button class="send-button" id="sendButton" onclick="sendMessage();" title="Send | إرسال">
                        ➤
                    </button>
                    <textarea 
                        class="message-input" 
                        id="messageInput" 
                        placeholder="Enter your legal inquiry in Arabic or English | اكتب استفسارك القانوني باللغة العربية أو الإنجليزية"
                        onkeydown="handleKeyPress(event)"
                        rows="1"
                    ></textarea>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isWaiting = false;
        let requestTimeout = null;
        let messageCount = 0;
        const MAX_MESSAGES = 50; // Prevent memory issues
        const MAX_MESSAGE_LENGTH = 5000; // Prevent UI breaking
        const REQUEST_TIMEOUT = 30000; // 30 seconds timeout
        
        // Debounce function to prevent rapid clicks
        function debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }
        
        // Sanitize HTML content
        function sanitizeHTML(str) {
            const temp = document.createElement('div');
            temp.textContent = str;
            return temp.innerHTML;
        }
        
        // Validate message content
        function validateMessage(message) {
            if (!message || typeof message !== 'string') {
                return { valid: false, error: 'Invalid message format' };
            }
            
            if (message.length === 0) {
                return { valid: false, error: 'Message cannot be empty' };
            }
            
            if (message.length > MAX_MESSAGE_LENGTH) {
                return { valid: false, error: `Message too long (max ${MAX_MESSAGE_LENGTH} characters)` };
            }
            
            return { valid: true };
        }
        
        // Clean up old messages to prevent memory issues
        function cleanupMessages() {
            const chatMessages = document.getElementById('messages');
            if (!chatMessages) return;
            
            const messages = chatMessages.querySelectorAll('.message');
            if (messages.length > MAX_MESSAGES) {
                const toRemove = messages.length - MAX_MESSAGES;
                for (let i = 0; i < toRemove; i++) {
                    if (messages[i]) {
                        messages[i].remove();
                    }
                }
            }
        }
        
        function addMessage(content, type) {
            try {
                const chatMessages = document.getElementById('messages');
                if (!chatMessages) {
                    return;
                }
                
                // Validate inputs
                if (!type || (type !== 'user' && type !== 'assistant')) {
                    return;
                }
                
                // Ensure content is a string and sanitize
                content = String(content || '').trim();
                if (!content) {
                    content = 'Empty message';
                }
                
                // Truncate very long messages
                if (content.length > MAX_MESSAGE_LENGTH) {
                    content = content.substring(0, MAX_MESSAGE_LENGTH) + '... (message truncated)';
                }
                
                // Clean up old messages first
                cleanupMessages();
                
                const welcomeSection = chatMessages.querySelector('.welcome-section');
                if (welcomeSection) {
                    welcomeSection.remove();
                }
                
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + type;
                
                let timeStr = '';
                try {
                    const now = new Date();
                    timeStr = now.toLocaleTimeString('ar-EG', { 
                        hour: '2-digit', 
                        minute: '2-digit' 
                    });
                } catch (e) {
                    // Fallback for browsers that don't support ar-EG locale
                    timeStr = new Date().toLocaleTimeString();
                }
                
                messageCount++;
            
            // Detect language for assistant messages
            let languageClass = '';
            if (type === 'assistant') {
                try {
                    // Count Arabic vs English characters  
                    const arabicChars = (content.match(/[\u0600-\u06FF]/g) || []).length;
                    const englishChars = (content.match(/[a-zA-Z]/g) || []).length;
                    const totalChars = arabicChars + englishChars;
                    
                    if (totalChars > 0 && englishChars / totalChars > 0.6) {
                        languageClass = ' english';
                    }
                } catch (e) {
                    languageClass = '';
                }
            }
            
            // Format content with better spacing
            let formattedContent = content;
            if (type === 'assistant') {
                try {
                    // Add proper spacing around numbered items and bullet points
                    formattedContent = formattedContent
                        .replace(/(\\n)(\\d+\\. )/g, '\\n\\n$2') // Add space before numbered items
                        .replace(/(\\d+\\. [^\\n]+)(\\n-)/g, '$1\\n$2') // Add space after numbered titles
                        .replace(/(\\n)(- )/g, '\\n$2'); // Ensure bullet points have proper spacing
                } catch (e) {
                    formattedContent = content;
                }
            }
            
                messageDiv.innerHTML = 
                    '<div class="message-bubble' + languageClass + '">' + formattedContent + '</div>' +
                    '<div class="message-time">' + timeStr + '</div>';
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } catch (error) {
                // Fallback: add simple message
                const chatMessages = document.getElementById('messages');
                if (chatMessages) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message ' + type;
                    messageDiv.innerHTML = '<div class="message-bubble">' + String(content || 'Error displaying message') + '</div>';
                    chatMessages.appendChild(messageDiv);
                }
            }
        }
        
        function showTyping() {
            try {
                const chatMessages = document.getElementById('messages');
                if (!chatMessages) return;
                
                const typingDiv = document.createElement('div');
                typingDiv.className = 'typing-indicator';
                typingDiv.id = 'typingIndicator';
                
                typingDiv.innerHTML = 
                    '<div class="typing-bubble">' +
                        '<span>🔍 Analyzing legal documents | جاري تحليل الوثائق القانونية...</span>' +
                        '<div class="typing-dots">' +
                            '<span></span>' +
                            '<span></span>' +
                            '<span></span>' +
                        '</div>' +
                    '</div>';
                chatMessages.appendChild(typingDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } catch (error) {
                // Silent error handling
            }
        }
        
        function hideTyping() {
            try {
                const typingIndicator = document.getElementById('typingIndicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            } catch (error) {
                // Silent error handling
            }
        }
        
        async function sendMessage() {
            try {
                const messageInput = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');
                
                if (!messageInput || !sendButton) {
                    return;
                }
                
                const message = messageInput.value.trim();
                
                // Validate message
                const validation = validateMessage(message);
                if (!validation.valid) {
                    addMessage(`❌ ${validation.error}`, 'assistant');
                    return;
                }
                
                if (isWaiting) {
                    return;
                }
                
                // Clear any existing timeout
                if (requestTimeout) {
                    clearTimeout(requestTimeout);
                }
                
                addMessage(message, 'user');
                messageInput.value = '';
                
                isWaiting = true;
                sendButton.disabled = true;
                sendButton.innerHTML = '⏳';
                showTyping();
                
                // Set request timeout
                requestTimeout = setTimeout(() => {
                    hideTyping();
                    addMessage('❌ Request timeout. Please try again. | انتهت مهلة الطلب. يرجى المحاولة مرة أخرى.', 'assistant');
                    resetSendButton(messageInput, sendButton);
                }, REQUEST_TIMEOUT);
                
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
                
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({query: message}),
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                clearTimeout(requestTimeout);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                let data;
                try {
                    data = await response.json();
                } catch (e) {
                    throw new Error('Invalid JSON response from server');
                }
                
                hideTyping();
                
                if (data && typeof data === 'object') {
                    if (data.success && data.result && data.result.answer) {
                        addMessage(data.result.answer, 'assistant');
                    } else {
                        const errorMsg = data.error || 'Unknown server error';
                        addMessage(`❌ Error | خطأ: ${errorMsg}`, 'assistant');
                    }
                } else {
                    addMessage('❌ Invalid response from server | استجابة غير صحيحة من الخادم', 'assistant');
                }
                
            } catch (error) {
                hideTyping();
                
                if (requestTimeout) {
                    clearTimeout(requestTimeout);
                }
                
                let errorMessage = '❌ Connection Error | خطأ في الاتصال';
                
                if (error.name === 'AbortError') {
                    errorMessage = '❌ Request was cancelled | تم إلغاء الطلب';
                } else if (error.message.includes('timeout')) {
                    errorMessage = '❌ Request timeout | انتهت مهلة الطلب';
                } else if (error.message.includes('network')) {
                    errorMessage = '❌ Network error | خطأ في الشبكة';
                } else if (error.message) {
                    errorMessage += ': ' + error.message;
                }
                
                addMessage(errorMessage, 'assistant');
            } finally {
                const messageInput = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');
                resetSendButton(messageInput, sendButton);
            }
        }
        
        // Helper function to reset send button state
        function resetSendButton(messageInput, sendButton) {
            isWaiting = false;
            if (sendButton) {
                sendButton.disabled = false;
                sendButton.innerHTML = '➤';
            }
            if (messageInput) {
                messageInput.focus();
            }
        }
        
        // Debounced version of sendMessage to prevent rapid clicks
        const debouncedSendMessage = debounce(sendMessage, 300);
        
        function askQuestion(query) {
            try {
                const messageInput = document.getElementById('messageInput');
                if (!messageInput) {
                    return;
                }
                
                if (isWaiting) {
                    return;
                }
                
                const validation = validateMessage(query);
                if (!validation.valid) {
                    addMessage(`❌ ${validation.error}`, 'assistant');
                    return;
                }
                
                messageInput.value = query;
                debouncedSendMessage();
            } catch (error) {
                // Silent error handling
            }
        }
        
        function handleKeyPress(event) {
            try {
                
                if (!event || !event.key) {
                    return true;
                }
                
                if (event.key === 'Enter') {
                    if (event.shiftKey) {
                        // Shift + Enter: Allow new line (default behavior)
                        return true;
                    } else {
                        // Enter alone: Send message
                        if (event.preventDefault) {
                            event.preventDefault();
                        }
                        
                        if (!isWaiting) {
                            debouncedSendMessage();
                        }
                        return false;
                    }
                }
                
                return true;
            } catch (error) {
                return true;
            }
        }
        
        // Auto-resize textarea and focus input on load
        window.onload = function() {
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            
            if (messageInput) {
                messageInput.focus();
                
                // Auto-resize textarea
                messageInput.addEventListener('input', function() {
                    this.style.height = 'auto';
                    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
                });
            }
        }
        
        // Clear chat history function
        async function clearChatHistory() {
            try {
                const confirmMessage = `هل تريد بدء محادثة جديدة؟ سيتم مسح تاريخ المحادثة الحالي.

Do you want to start a new chat? This will clear the current conversation history.`;
                
                const confirmed = confirm(confirmMessage);
                
                if (!confirmed) return;
                
                const response = await fetch('/clear-history', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                if (response.ok) {
                    // Clear the messages container
                    const chatMessages = document.getElementById('messages');
                    if (chatMessages) {
                        chatMessages.innerHTML = `
                            <div class="welcome-section">
                                <p class="notice">
                                    <span class="arabic-text">⚠️ تنبيه مهم: هذا النظام للمعلومات العامة فقط وليس بديلاً عن الاستشارة القانونية المهنية.</span><br/>
                                    <span class="english-text">⚠️ Important Notice: This system provides general information only and is not a substitute for professional legal advice.</span>
                                </p>
                            </div>
                            
                            <div class="sample-questions">
                                <button class="sample-question" onclick="askQuestion('ما هي شروط تأسيس الشركات؟')">شروط تأسيس الشركات 🏢</button>
                                <button class="sample-question" onclick="askQuestion('ما هي حقوق المستهلك في الأردن؟')">حقوق المستهلك 📍</button>
                                <button class="sample-question" onclick="askQuestion('كيف أسجل علامة تجارية؟')">تسجيل العلامات التجارية ®</button>
                                <button class="sample-question" onclick="askQuestion('ما هي قوانين الاستثمار الأجنبي؟')">قوانين الاستثمار 💰</button>
                                <button class="sample-question" onclick="askQuestion('إجراءات التسجيل التجاري')">التسجيل التجاري 📋</button>
                                <button class="sample-question" onclick="askQuestion('قوانين العمل والعمال')">قوانين العمل 👔</button>
                            </div>
                            
                            <div style="text-align: center; margin-top: 10px;">
                                <button 
                                    onclick="clearChatHistory()" 
                                    style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171; padding: 6px 12px; border-radius: 8px; font-size: 0.8em; cursor: pointer;"
                                    title="Clear conversation history | مسح تاريخ المحادثة"
                                >
                                    🗑️ بدء محادثة جديدة | New Chat
                                </button>
                            </div>
                        `;
                    }
                    
                    messageCount = 0;
                } else {
                    // Failed to clear chat history
                }
                
            } catch (error) {
                // Silent error handling
            }
        }
        
        // Global error handler
        window.onerror = function(message, source, lineno, colno, error) {
            // Silent error handling for production
            return false;
        };
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Main page"""
    return render_template_string(ADVANCED_HTML_TEMPLATE)

@app.route('/query', methods=['POST'])
def process_query():
    """Process legal query using advanced system with input validation"""
    try:
        if not legal_system:
            return jsonify({
                'success': False,
                'error': 'النظام في وضع محدود - لم يتم تحميل قاعدة البيانات بعد. يرجى تحميل ملفات ChromaDB أولاً. | System in limited mode - database not loaded yet. Please upload ChromaDB files first.'
            })
        
        data = request.get_json()
        query = data.get('query', '').strip()
        
        print(f"🔍 Received query: {query}")
        
        # Get session for conversation history first (needed for validation)
        session_id = get_or_create_session()
        
        # Get full conversation context (both user and assistant messages)
        conversation_context = get_conversation_context(session_id, limit=3)
        
        # Validate and filter query before processing
        validation_result = validate_and_filter_query(query, conversation_context)
        
        if not validation_result['valid']:
            print(f"❌ Invalid query: {validation_result['response']}")
            return jsonify({
                'success': False,
                'error': validation_result['response']
            })
        
        # If query should not be processed (inappropriate, greeting, non-legal)
        if not validation_result['should_process']:
            print(f"🛑 Query filtered: returning canned response")
            
            # Add to conversation history (session_id already retrieved above)
            add_to_conversation(session_id, 'user', query)
            add_to_conversation(session_id, 'assistant', validation_result['response'], {
                'confidence': 1.0,
                'query_type': 'filtered',
                'processing_time': 0.01,
                'sources_count': 0
            })
            
            return jsonify({
                'success': True,
                'result': {
                    'answer': validation_result['response'],
                    'confidence': 1.0,
                    'query_type': 'filtered',
                    'processing_time': 0.01,
                    'sources': [],
                    'citations': [],
                    'reasoning_steps': ['Query filtered by input validation']
                }
            })
        
        # Get conversation history for the RAG system (session_id and conversation_context already retrieved above)
        conversation_history = []
        if session_id in conversation_store:
            recent_messages = conversation_store[session_id][-6:]  # Last 6 messages
            conversation_history = [msg['content'] for msg in recent_messages if msg['type'] == 'user']
        
        print(f"🔍 Processing legal query: {query}")
        
        # Check if this is a follow-up question
        is_follow_up = is_follow_up_question(query, conversation_context)
        
        if is_follow_up and conversation_context:
            print(f"🔄 Detected follow-up question with context")
            # For follow-up questions, create a more explicit context-aware query
            enriched_query = f"""This is a follow-up question referencing a previous response. Please answer based on the previous conversation context.

PREVIOUS CONVERSATION CONTEXT:
{conversation_context}

CURRENT FOLLOW-UP QUESTION: {query}

Instructions: If the user is asking about a specific point, step, or element from the previous response, provide detailed information about that specific item rather than generating a new general response. Reference the exact content from the previous conversation."""
            
            result: QueryResult = legal_system.query(enriched_query, conversation_history)
        else:
            # Regular query processing
            result: QueryResult = legal_system.query(query, conversation_history)
        
        # Apply our improved text cleaning to RAG responses too
        result.answer = clean_response_formatting(result.answer)
        
        # Add to conversation history
        add_to_conversation(session_id, 'user', query)
        add_to_conversation(session_id, 'assistant', result.answer, {
            'confidence': result.confidence,
            'query_type': result.query_type,
            'processing_time': result.processing_time,
            'sources_count': len(result.sources)
        })
        
        print(f"✅ Query processed successfully. Confidence: {result.confidence:.2f}")
        
        return jsonify({
            'success': True,
            'result': {
                'answer': result.answer,
                'confidence': result.confidence,
                'query_type': result.query_type,
                'processing_time': result.processing_time,
                'sources': result.sources,
                'citations': result.citations,
                'reasoning_steps': result.reasoning_steps
            }
        })
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return jsonify({
            'success': False,
            'error': f'حدث خطأ في معالجة الاستفسار: {str(e)}'
        })

@app.route('/stats')
def get_stats():
    """Get system statistics"""
    if not legal_system:
        return jsonify({'error': 'System not initialized'})
    
    return jsonify(legal_system.get_system_stats())

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'system_ready': legal_system is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/history')
def get_conversation_history():
    """Get conversation history for current session"""
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in conversation_store:
            return jsonify({'success': True, 'history': []})
        
        history = conversation_store[session_id]
        return jsonify({
            'success': True, 
            'history': history,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear-history', methods=['POST'])
def clear_conversation_history():
    """Clear conversation history for current session"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in conversation_store:
            conversation_store[session_id] = []
        
        return jsonify({'success': True, 'message': 'History cleared'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Advanced Legal RAG Web Demo...")
    
    # Initialize the system (will always succeed now, even without data)
    initialize_system()
    
    print("🌐 Starting web server...")
    if legal_system:
        print("📊 System ready with advanced features:")
        print("   • Hybrid semantic + keyword search")
        print("   • Query type classification")
        print("   • Multi-step reasoning")
        print("   • Confidence scoring")
        print("   • Source attribution")
        print("   • Conversation memory")
    else:
        print("📊 System started in limited mode:")
        print("   • Web interface available")
        print("   • Upload ChromaDB data to enable full functionality")
    print()
    
    # Use PORT environment variable for production deployment (Render, Heroku, etc.)
    import os
    port = int(os.environ.get('PORT', 5003))
    print(f"🌐 Starting server on port {port}")
    print("⚠️ Make sure OPENAI_API_KEY is set in environment")
    
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True) 