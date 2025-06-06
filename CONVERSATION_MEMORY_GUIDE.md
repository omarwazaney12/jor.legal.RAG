# 🧠 Conversation Memory Guide

## Overview

The Legal RAG System now has advanced conversation memory that allows users to have natural, context-aware conversations with follow-up questions and references to previous responses.

## 🎯 Key Features

### 1. **Follow-up Question Detection**
The system automatically detects when users ask follow-up questions by recognizing keywords like:

**Arabic Keywords:**
- `هذا`, `ذلك`, `هذه`, `تلك` (this, that)
- `المذكور`, `السابق`, `المشار إليه` (mentioned, previous, referred to)
- `أخبرني أكثر`, `وضح`, `اشرح`, `تفصيل` (tell me more, explain, detail)
- `النقطة الأولى/الثانية/الثالثة` (first/second/third point)
- `البند الأول/الثاني` (first/second clause)
- `الخطوة الأولى/الثانية` (first/second step)
- `وماذا عن`, `وما هو`, `وما هي` (and what about, and what is)

**English Keywords:**
- "this", "that", "mentioned", "previous"
- "tell me more", "explain", "clarify"
- "first point", "second step"
- "what about", "regarding"

### 2. **Context-Aware Responses**
When a follow-up question is detected, the system:
- Includes previous conversation context in the prompt
- References earlier responses when generating new answers
- Maintains topic continuity across multiple questions
- Provides detailed explanations building on previous context

### 3. **Conversation History Storage**
- **Session-based**: Each browser session maintains its own conversation history
- **Memory limit**: Keeps last 20 messages to prevent memory issues
- **Context window**: Uses last 3 Q&A pairs for context awareness
- **Automatic cleanup**: Older messages are automatically removed

## 🚀 How to Use

### Basic Conversation Flow

1. **Start with a general question:**
   ```
   ما هي شروط تأسيس الشركات في الأردن؟
   ```

2. **Ask follow-up questions:**
   ```
   ما هو الحد الأدنى لرأس المال المطلوب؟
   أخبرني أكثر عن النقطة الثانية
   وما هي الوثائق المطلوبة؟
   ```

3. **Reference previous responses:**
   ```
   وضح لي هذا الإجراء بالتفصيل
   ما هي المدة المحددة لذلك؟
   ```

### Advanced Usage Examples

#### Example 1: Company Formation
```
User: ما هي شروط تأسيس الشركات؟
System: [Lists numbered requirements]

User: أخبرني أكثر عن النقطة الثالثة
System: [Detailed explanation of the third requirement with legal references]

User: وما هي الجهة المختصة بهذا الإجراء؟
System: [Explains the competent authority based on previous context]
```

#### Example 2: Consumer Rights
```
User: ما هي حقوق المستهلك في الأردن؟
System: [Lists consumer rights]

User: كيف أشتكي على متجر؟
System: [Explains complaint procedures]

User: ما هي المدة المحددة لهذا الإجراء؟
System: [Explains timeframes for complaints based on previous context]
```

## 🛠️ Technical Implementation

### 1. Conversation Context Function
```python
def get_conversation_context(session_id: str, limit: int = 3) -> str:
    """Get recent conversation history for context-aware responses"""
    # Returns formatted context from last few Q&A pairs
```

### 2. Follow-up Detection
```python
def is_follow_up_question(query: str, conversation_context: str) -> bool:
    """Detect if this is a follow-up question referencing previous conversation"""
    # Checks for follow-up keywords and conversation context
```

### 3. Context-Enhanced Prompts
When a follow-up is detected, the system enriches the query:
```python
enriched_query = f"السياق السابق: {conversation_context}\n\nالسؤال الحالي: {query}"
```

### 4. Session Management
- Uses Flask sessions with UUID-based session IDs
- Stores conversation history in memory (can be replaced with database)
- Automatic session creation and cleanup

## 📊 API Endpoints

### Query with Memory
```http
POST /query
Content-Type: application/json

{
    "query": "أخبرني أكثر عن النقطة الثانية"
}
```

### Get History
```http
GET /history
```

### Clear History
```http
POST /clear-history
```

## 🧪 Testing

Use the provided test script:
```bash
python conversation_memory_test.py
```

This script tests:
- Initial questions and follow-ups
- Numbered point references
- Topic switching and returning
- English conversation memory
- History management

## 🎨 UI Features

### Chat Interface
- **Follow-up indicators**: Shows when context is being used
- **History display**: Users can see their conversation flow
- **Clear history button**: Start fresh conversations
- **Typing indicators**: Shows when context is being processed

### Sample Questions
The interface includes sample questions that demonstrate conversation flow:
- Initial questions to start conversations
- Follow-up examples for context testing

## 💡 Best Practices

### For Users
1. **Start with clear questions**: Begin with specific legal topics
2. **Use reference words**: Use "هذا", "ذلك", "النقطة الثانية" for follow-ups
3. **Be specific**: "وضح الخطوة الثالثة" is better than just "وضح"
4. **Topic transitions**: Clearly indicate when switching topics

### For Developers
1. **Context limits**: Keep conversation context under 500 characters
2. **Session cleanup**: Implement proper session cleanup for production
3. **Error handling**: Handle cases where context is incomplete
4. **Performance**: Monitor response times with large conversation histories

## 🔧 Configuration

### Memory Settings
```python
# Maximum messages per session
MAX_MESSAGES = 20

# Context window for follow-ups
CONTEXT_LIMIT = 3

# Context character limit
MAX_CONTEXT_LENGTH = 300
```

### Follow-up Detection
```python
# Customize follow-up keywords
follow_up_keywords = [
    'هذا', 'ذلك', 'النقطة الثانية', 
    'أخبرني أكثر', 'وضح'
]
```

## 🚨 Limitations

1. **Memory scope**: Only remembers within same session
2. **Context length**: Limited context to prevent token overflow
3. **Topic mixing**: May confuse context if topics are mixed rapidly
4. **Language switching**: Works best when staying in same language

## 🔮 Future Enhancements

1. **Persistent memory**: Store conversations in database
2. **User accounts**: Personal conversation history
3. **Smart summaries**: Compress long conversations
4. **Topic modeling**: Better topic switching detection
5. **Multi-session context**: Reference conversations across sessions

## 📝 Examples in Action

### Real Conversation Flow
```
🗣️ User: ما هي شروط تأسيس شركة تجارية؟

🤖 System: شروط تأسيس الشركة التجارية في الأردن:

1. رأس المال المطلوب:
   - الحد الأدنى 1000 دينار أردني
   - يجب إيداع 25% عند التأسيس
   
2. الوثائق المطلوبة:
   - عقد التأسيس
   - النظام الداخلي
   - شهادة عدم محكومية
   
3. الإجراءات القانونية:
   - التسجيل في دائرة مراقب الشركات
   - الحصول على رخصة تجارية

---

🗣️ User: أخبرني أكثر عن النقطة الثانية

🤖 System: [Building on previous context about required documents]

الوثائق المطلوبة للتأسيس (التفصيل الكامل):

1. عقد التأسيس:
   - يجب أن يكون مصدقاً من كاتب العدل
   - يحدد أسماء الشركاء ونسب مساهماتهم
   
2. النظام الداخلي:
   - ينظم إدارة الشركة وصلاحيات المدراء
   - يحدد آلية اتخاذ القرارات
   
[Context-aware response continues...]

---

🗣️ User: وما هي الرسوم المطلوبة؟

🤖 System: [Referencing company formation context]

الرسوم المطلوبة لتأسيس الشركة التجارية:

1. رسوم التسجيل:
   - رسم تسجيل ثابت: 100 دينار
   - رسم متغير حسب رأس المال: 0.5%
   
[Continues with relevant fee structure...]
```

This conversation shows how the system maintains context across multiple questions, providing detailed, relevant answers that build on previous responses. 