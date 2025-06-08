##!/usr/bin/env python3
"""
Advanced Legal RAG System for Jordanian Laws
Handles 101+ legal documents with sophisticated retrieval and reasoning
"""

import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime
import hashlib
import time
import pickle

# Core libraries
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb
from chromadb.config import Settings

# ML and NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# For retry logic
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1, max_delay=60, backoff_factor=2):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Check for rate limiting
                    if "rate limit" in error_msg or "quota" in error_msg:
                        delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                        print(f"   ⏳ Rate limit hit. Waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(delay)
                        continue
                    
                    # Check for network/connection errors
                    elif any(term in error_msg for term in ["connection", "timeout", "network", "unreachable"]):
                        delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                        print(f"   🌐 Network error. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    
                    # For other errors, don't retry immediately
                    else:
                        if attempt < max_retries:
                            print(f"   ⚠️  Error: {e}. Retrying... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(1)
                        else:
                            print(f"   ❌ Failed after {max_retries} retries: {e}")
                            break
            
            raise last_exception
        return wrapper
    return decorator

@dataclass
class LegalDocument:
    """Enhanced document representation"""
    id: str
    title: str
    source_file: str
    content: str
    doc_type: str  # "قانون", "نظام", "تعليمات"
    category: str  # Legal domain
    law_number: Optional[str]
    year: Optional[str]
    articles: List[Dict[str, str]]  # Extracted articles
    metadata: Dict[str, Any]
    word_count: int
    language: str = "arabic"

@dataclass
class QueryResult:
    """Enhanced query result with confidence and reasoning"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    reasoning_steps: List[str]
    citations: List[str]
    query_type: str
    processing_time: float

class AdvancedLegalProcessor:
    """Advanced document processor with legal-specific intelligence"""
    
    def __init__(self):
        self.article_patterns = [
            r'المادة\s*\(?(\d+)\)?',
            r'البند\s*\(?(\d+)\)?', 
            r'الفقرة\s*\(?(\d+)\)?',
            r'البـــــاب\s*\(?(\d+)\)?'
        ]
        
        self.law_metadata_patterns = {
            'law_number': r'رقم\s*\(?(\d+)\)?\s*لسنة\s*(\d+)',
            'ministry': r'وزارة\s+([\u0600-\u06FF\s]+)',
            'effective_date': r'يعمل\s+به\s+.*?(\d{4})',
            'published_date': r'نشره?\s+في\s+الجريدة\s+الرسمية'
        }
        
    def normalize_arabic_number(self, number: str) -> str:
        """Normalize Arabic and Arabic-Indic numerals to standard format"""
        # Arabic-Indic to Arabic numerals mapping
        arabic_indic_to_arabic = {
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }
        
        # Convert Arabic-Indic numerals to Arabic
        normalized = number
        for arabic_indic, arabic in arabic_indic_to_arabic.items():
            normalized = normalized.replace(arabic_indic, arabic)
        
        # Remove leading zeros and normalize
        normalized = re.sub(r'^0+', '', normalized)
        if not normalized or not normalized.isdigit():  # If all zeros or non-numeric
            normalized = '0'
        
        return normalized

    def extract_articles(self, content: str) -> List[Dict[str, str]]:
        """Extract individual articles from legal text with deduplication"""
        articles = []
        seen_articles = set()  # Track normalized numbers to avoid duplicates
        
        # Enhanced patterns to catch various article formats
        patterns = [
            r'المادة\s*\(?([٠-٩0-9]+)\)?\s*:?\s*',
            r'البند\s*\(?([٠-٩0-9]+)\)?\s*:?\s*',
            r'الفقرة\s*\(?([٠-٩0-9]+)\)?\s*:?\s*'
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE))
            
            for i, match in enumerate(matches):
                raw_number = match.group(1)
                normalized_number = self.normalize_arabic_number(raw_number)
                
                # Skip if we've already seen this normalized number
                if normalized_number in seen_articles:
                    continue
                    
                seen_articles.add(normalized_number)
                
                # Extract article content until next article
                start_pos = match.end()
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                    article_content = content[start_pos:end_pos].strip()
                else:
                    article_content = content[start_pos:].strip()
                
                # Clean up content
                article_content = re.sub(r'\n+', ' ', article_content)
                article_content = article_content.strip()
                
                if len(article_content) > 10:  # Only add if has substantial content
                    article_title = article_content[:100] + '...' if len(article_content) > 100 else article_content
                    
                    articles.append({
                        'number': normalized_number,
                        'title': article_title,
                        'content': article_content,
                        'full_text': f"المادة {normalized_number}: {article_content}"
                    })
        
        # Sort by numeric value
        return sorted(articles, key=lambda x: int(x['number']) if x['number'].isdigit() else 999)
    
    def extract_legal_metadata(self, title: str, content: str) -> Dict[str, Any]:
        """Extract comprehensive legal metadata"""
        metadata = {
            'extracted_at': datetime.now().isoformat(),
            'title_cleaned': self.clean_title(title)
        }
        
        # Extract law number and year
        law_match = re.search(self.law_metadata_patterns['law_number'], content)
        if law_match:
            metadata['law_number'] = law_match.group(1)
            metadata['year'] = law_match.group(2)
        
        # Extract ministry
        ministry_match = re.search(self.law_metadata_patterns['ministry'], content)
        if ministry_match:
            metadata['ministry'] = ministry_match.group(1).strip()
        
        # Document statistics
        metadata['total_articles'] = len(re.findall(r'المادة\s*\d+', content))
        metadata['total_chapters'] = len(re.findall(r'الباب\s*\d+', content))
        metadata['total_items'] = len(re.findall(r'البند\s*\d+', content))
        
        # Legal terms frequency
        legal_terms = ['حقوق', 'واجبات', 'التزامات', 'مسؤوليات', 'عقوبات', 
                      'مخالفات', 'رسوم', 'ضرائب', 'ترخيص', 'تسجيل', 'موافقة']
        
        metadata['legal_terms_found'] = []
        for term in legal_terms:
            if term in content:
                metadata['legal_terms_found'].append(term)
        
        return metadata
    
    def clean_title(self, title: str) -> str:
        """Clean and standardize document titles"""
        # Remove file extensions and common prefixes
        cleaned = title.replace('.txt', '').replace('_', ' ').replace('-', ' ')
        cleaned = re.sub(r'^(laws_|systems_|instructions_)', '', cleaned)
        return cleaned.strip()
    
    def categorize_document(self, title: str, content: str) -> Tuple[str, str]:
        """Advanced document categorization"""
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Document type classification
        if title.startswith('laws_') or 'قانون' in title_lower:
            doc_type = "قانون"
        elif title.startswith('systems_') or 'نظام' in title_lower:
            doc_type = "نظام"
        elif title.startswith('instructions_') or 'تعليمات' in title_lower:
            doc_type = "تعليمات"
        elif 'الدستور' in title_lower:
            doc_type = "دستور"
        else:
            doc_type = "متنوعة"
        
        # Legal domain classification
        domain_keywords = {
            'حماية المستهلك': ['مستهلك', 'حماية المستهلك', 'المستهلكين'],
            'قانون الشركات': ['شركة', 'شركات', 'مساهمة', 'محدودة'],
            'تجارة خارجية': ['استيراد', 'تصدير', 'جمارك', 'تعرفة'],
            'تسجيل تجاري': ['تسجيل تجاري', 'السجل التجاري'],
            'استثمار': ['استثمار', 'الاستثمار', 'استثمارية'],
            'عمل وضمان': ['عمل', 'عمال', 'ضمان اجتماعي'],
            'بيئة': ['بيئة', 'بيئي', 'تلوث'],
            'مالية وضرائب': ['ضريبة', 'ضرائب', 'رسوم', 'مالية'],
            'ملكية فكرية': ['براءة', 'علامة تجارية', 'ملكية فكرية']
        }
        
        category = "عام"  # Default
        max_matches = 0
        
        for domain, keywords in domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in content_lower or keyword in title_lower)
            if matches > max_matches:
                max_matches = matches
                category = domain
        
        return doc_type, category

class AdvancedVectorStore:
    """Advanced vector storage with hybrid search capabilities"""
    
    def __init__(self, collection_name: str = "jordanian_legal_docs"):
        self.collection_name = collection_name
        
        # Initialize embeddings as None - will be created when needed for queries
        self.embeddings = None
        
        # Initialize ChromaDB with in-memory client (no SQLite issues)
        self.chroma_client = chromadb.Client()  # In-memory client
        
        # Progress tracking (for backward compatibility)
        self.progress_file = Path("embedding_progress.pkl")
        self.error_log_file = Path("embedding_errors.log")
        
        # Initialize collection as None - will be loaded from pre-built data
        self.collection = None
        
        # TF-IDF for keyword search
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Keep Arabic stopwords for legal precision
            ngram_range=(1, 3)
        )
        self.tfidf_matrix = None
        self.doc_chunks = []
    
    def clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata by removing None values and ensuring ChromaDB compatibility"""
        cleaned = {}
        for key, value in metadata.items():
            if value is not None:
                # Convert all values to strings, ints, floats, or bools for ChromaDB compatibility
                if isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                else:
                    cleaned[key] = str(value)
        return cleaned
    
    def save_progress(self, completed_chunks: int, total_chunks: int, embeddings: List, chunk_metadata: List):
        """Save progress to resume later if needed"""
        progress_data = {
            'completed_chunks': completed_chunks,
            'total_chunks': total_chunks,
            'embeddings': embeddings,
            'chunk_metadata': chunk_metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'wb') as f:
            pickle.dump(progress_data, f)
        print(f"   💾 Progress saved: {completed_chunks}/{total_chunks} chunks processed")
    
    def load_progress(self) -> Optional[Dict]:
        """Load previous progress if available"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"   ⚠️  Could not load progress: {e}")
                return None
        return None
    
    def log_error(self, error: str, chunk_index: int = None):
        """Log errors to file for debugging"""
        timestamp = datetime.now().isoformat()
        error_entry = f"[{timestamp}] Chunk {chunk_index}: {error}\n"
        
        with open(self.error_log_file, 'a', encoding='utf-8') as f:
            f.write(error_entry)
    
    @retry_with_backoff(max_retries=5, base_delay=2, max_delay=120)
    def generate_embeddings_batch(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings with retry logic"""
        return self.embeddings.embed_documents(chunks)
    
    def clear_collection(self):
        """Clear the existing collection to rebuild with clean metadata"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            print(f"🗑️  Cleared existing collection: {self.collection_name}")
        except Exception as e:
            print(f"ℹ️  No existing collection to clear: {e}")
        
        # Recreate collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "Jordanian Legal Documents"}
        )
        print(f"✅ Created fresh collection: {self.collection_name}")

    def load_prebuilt_embeddings(self):
        """Load pre-built embeddings from JSON file"""
        print("📖 Loading pre-built embeddings...")
        
        # Try multiple possible paths for the embeddings file
        possible_paths = [
            Path("railway_embeddings_data/embeddings_data.json"),
            Path("./railway_embeddings_data/embeddings_data.json"),
            Path("/app/railway_embeddings_data/embeddings_data.json"),
            Path("embeddings_data.json")
        ]
        
        embeddings_file = None
        for path in possible_paths:
            print(f"🔍 Checking path: {path}")
            if path.exists():
                embeddings_file = path
                print(f"✅ Found embeddings at: {path}")
                break
        
        if not embeddings_file:
            print("❌ Pre-built embeddings not found in any of these locations:")
            for path in possible_paths:
                print(f"   - {path}")
            print("📁 Current working directory contents:")
            import os
            print(f"   Working dir: {os.getcwd()}")
            for item in os.listdir("."):
                print(f"   - {item}")
            return False
        
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"📊 Loaded {len(data['documents'])} pre-built documents")
            
            # Create collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Jordanian legal documents for RAG system",
                    "model": data.get('model', 'text-embedding-ada-002'),
                    "total_docs": data.get('total_docs', len(data['documents']))
                }
            )
            
            # Add all data to ChromaDB
            print("🔄 Adding documents to ChromaDB...")
            self.collection.add(
                documents=data['documents'],
                metadatas=data['metadatas'],
                embeddings=data['embeddings'],
                ids=data['ids']
            )
            
            # Build TF-IDF index for hybrid search
            print("🔍 Building TF-IDF index for hybrid search...")
            self.doc_chunks = data['documents']
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.doc_chunks)
            
            final_count = self.collection.count()
            print(f"✅ Successfully loaded {final_count} documents into ChromaDB")
            print(f"✅ TF-IDF index ready with {len(self.doc_chunks)} chunks")
            
            return True
                
        except Exception as e:
            print(f"❌ Error loading pre-built embeddings: {e}")
            return False

    def add_documents(self, documents: List[LegalDocument], chunk_size: int = 500, force_rebuild: bool = False):
        """Load documents using pre-built embeddings"""
        print(f"📄 Using pre-built embeddings for {len(documents)} documents...")
        
        # Try to load pre-built embeddings first
        if self.load_prebuilt_embeddings():
            return
        
        # Fallback: If pre-built embeddings not available, show error
        print("❌ Pre-built embeddings not available. Please ensure railway_embeddings_data/embeddings_data.json exists.")
        print("🔧 Run build_raw_embeddings.py to generate embeddings first.")
    
    def _get_embeddings(self):
        """Lazy initialization of embeddings only when needed"""
        if self.embeddings is None:
            # Use raw OpenAI client instead of LangChain wrapper to avoid Railway proxies issue
            import openai
            self.embeddings = openai  # Store the module for direct API calls
        return self.embeddings
    
    def hybrid_search(self, query: str, top_k: int = 20, semantic_weight: float = 0.7) -> List[Dict]:
        """Advanced hybrid search combining semantic and keyword matching"""
        
        # 1. Semantic search using embeddings (lazy init - raw OpenAI API)
        openai_client = self._get_embeddings()
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"⚠️ Embedding generation failed: {e}")
            # Fallback: return empty results if embeddings fail
            return []
        
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Get more for reranking
            include=['documents', 'metadatas', 'distances']
        )
        
        # 2. Keyword search using TF-IDF
        keyword_scores = []
        if self.tfidf_matrix is not None and len(self.doc_chunks) > 0:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            keyword_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
            keyword_scores = keyword_similarities.tolist()  # Convert to list to avoid numpy array issues
        
        # 3. Combine scores and rerank
        combined_results = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            semantic_results['documents'][0],
            semantic_results['metadatas'][0], 
            semantic_results['distances'][0]
        )):
            # Convert distance to similarity (lower distance = higher similarity)
            semantic_score = max(0, 1 - distance)
            
            # Get keyword score for this document
            keyword_score = 0
            if keyword_scores and i < len(keyword_scores):
                keyword_score = float(keyword_scores[i])  # Ensure it's a float
            
            # Combined score
            final_score = (semantic_weight * semantic_score + 
                          (1 - semantic_weight) * keyword_score)
            
            combined_results.append({
                'content': doc,
                'metadata': metadata,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'final_score': final_score
            })
        
        # Sort by final score and return top_k
        combined_results.sort(key=lambda x: x['final_score'], reverse=True)
        return combined_results[:top_k]

class QueryClassifier:
    """Classify legal queries to optimize retrieval strategy"""
    
    def __init__(self):
        self.query_types = {
            'procedure': ['إجراءات', 'خطوات', 'كيفية', 'طريقة', 'مراحل'],
            'requirements': ['شروط', 'متطلبات', 'ضوابط', 'معايير'],
            'rights': ['حقوق', 'حق', 'مكفولة', 'مضمونة'],
            'obligations': ['واجبات', 'التزامات', 'مسؤوليات'],
            'penalties': ['عقوبات', 'غرامة', 'مخالفة', 'جزاء'],
            'definitions': ['تعريف', 'معنى', 'المقصود', 'يقصد'],
            'establishment': ['إنشاء', 'تأسيس', 'فتح', 'إقامة'],
            'registration': ['تسجيل', 'قيد', 'تصريح', 'ترخيص']
        }
    
    def classify(self, query: str) -> str:
        """Classify query type for optimized retrieval"""
        query_lower = query.lower()
        
        type_scores = {}
        for query_type, keywords in self.query_types.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                type_scores[query_type] = score
        
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        return 'general'

# Initialize the system
print("🚀 Advanced Legal RAG System Loading...")

class LegalReasoningEngine:
    """Advanced reasoning engine for complex legal queries"""
    
    def __init__(self, vector_store: AdvancedVectorStore):
        self.vector_store = vector_store
        self.query_classifier = QueryClassifier()
        
        # Reasoning templates for different query types
        self.reasoning_templates = {
            'procedure': self._procedure_reasoning,
            'requirements': self._requirements_reasoning,
            'establishment': self._establishment_reasoning,
            'registration': self._registration_reasoning,
            'general': self._general_reasoning
        }
    
    def detect_language(self, text: str) -> str:
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
    
    def clean_response_formatting(self, response: str) -> str:
        """Clean and standardize response formatting"""
        import re
        
        # Remove all markdown formatting
        cleaned = response
        
        # Remove bold formatting (**text**)
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
        
        # Remove italic formatting (*text*)
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
        
        # Remove markdown headers (## text)
        cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)
        
        # Standardize numbered list formatting
        # Pattern: "1. **Title:**" -> "1. Title:"
        cleaned = re.sub(r'^(\d+)\.\s*\*\*(.*?)\*\*:?\s*$', r'\1. \2:', cleaned, flags=re.MULTILINE)
        
        # Pattern: "1. Title:" (ensure colon at end)
        cleaned = re.sub(r'^(\d+)\.\s+([^:\n]+)(?::?)$', r'\1. \2:', cleaned, flags=re.MULTILINE)
        
        # Ensure bullet points use consistent dash formatting
        # Convert various bullet formats to standard dash
        cleaned = re.sub(r'^[\s]*[•·‧⁃▪▫‣⁌⁍]*\s*', '- ', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^[\s]*[\*\+]\s+', '- ', cleaned, flags=re.MULTILINE)
        
        # Fix multiple dashes to single dash
        cleaned = re.sub(r'^-+\s*', '- ', cleaned, flags=re.MULTILINE)
        
        # Ensure proper spacing after numbers
        cleaned = re.sub(r'^(\d+)\.([^\s])', r'\1. \2', cleaned, flags=re.MULTILINE)
        
        # Clean up excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        # Enhanced line spacing formatting
        lines = cleaned.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # Check if this is a numbered item (main title)
            is_numbered = re.match(r'^\d+\.\s+', line)
            # Check if this is a bullet point
            is_bullet = re.match(r'^-\s+', line)
            # Check if this is a reference line
            is_reference = 'المرجع' in line or 'اقتباس' in line or 'Legal Reference' in line or 'Quote:' in line
            
            # Add extra spacing before numbered items (except first one)
            if is_numbered and formatted_lines and formatted_lines[-1] != '':
                formatted_lines.append('')
                formatted_lines.append('')
            
            # Add spacing before bullet points that aren't immediately after numbered items
            elif is_bullet and formatted_lines:
                prev_line = formatted_lines[-1] if formatted_lines else ''
                if prev_line and not re.match(r'^\d+\.\s+', prev_line) and not prev_line == '':
                    formatted_lines.append('')
            
            # Add spacing before references
            elif is_reference and formatted_lines and formatted_lines[-1] != '':
                if not re.match(r'^-\s+', formatted_lines[-1]):
                    formatted_lines.append('')
            
            formatted_lines.append(line)
            
            # Add spacing after numbered titles (before bullet points)
            if is_numbered:
                formatted_lines.append('')
        
        # Join lines and clean up
        cleaned = '\n'.join(formatted_lines)
        
        # Remove trailing spaces and clean up final formatting
        cleaned = '\n'.join(line.rstrip() for line in cleaned.split('\n'))
        
        # Final cleanup of excessive spacing
        cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)
        
        # Ensure single newline at end
        cleaned = cleaned.strip() + '\n' if cleaned.strip() else ''
        
        return cleaned.strip()
    
    def process_query(self, query: str, conversation_history: List[str] = None) -> QueryResult:
        """Process complex legal query with multi-step reasoning"""
        start_time = datetime.now()
        
        # Step 1: Classify query type
        query_type = self.query_classifier.classify(query)
        print(f"🔍 Query classified as: {query_type}")
        
        # Step 2: Enhanced retrieval based on query type
        relevant_docs = self._enhanced_retrieval(query, query_type)
        
        # Step 3: Apply reasoning template
        reasoning_func = self.reasoning_templates.get(query_type, self._general_reasoning)
        result = reasoning_func(query, relevant_docs, conversation_history)
        
        # Step 4: Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        result.processing_time = processing_time
        result.query_type = query_type
        
        return result
    
    def _enhanced_retrieval(self, query: str, query_type: str, top_k: int = 10) -> List[Dict]:
        """Enhanced retrieval strategy based on query type"""
        
        # Adjust search parameters based on query type
        if query_type in ['procedure', 'requirements', 'establishment']:
            # For procedural queries, prefer recent regulations and instructions
            semantic_weight = 0.8  # Higher semantic matching
            top_k = 12  # Get more documents for comprehensive procedures
        elif query_type in ['definitions', 'rights']:
            # For definitions, prefer exact legal text matches
            semantic_weight = 0.5  # Balance semantic and keyword
            top_k = 6  # Fewer, more precise results
        else:
            semantic_weight = 0.7  # Default balance
        
        # Perform hybrid search
        results = self.vector_store.hybrid_search(
            query=query,
            top_k=top_k,
            semantic_weight=semantic_weight
        )
        
        # Filter and rerank based on query type
        if query_type == 'establishment':
            # Prioritize laws and systems over instructions for establishment queries
            results = self._rerank_for_establishment(results)
        elif query_type == 'procedure':
            # Prioritize instructions and systems for procedural queries
            results = self._rerank_for_procedures(results)
        
        return results
    
    def _rerank_for_establishment(self, results: List[Dict]) -> List[Dict]:
        """Rerank results for establishment queries"""
        for result in results:
            metadata = result['metadata']
            
            # Boost laws and systems
            if metadata.get('doc_type') == 'قانون':
                result['final_score'] *= 1.3
            elif metadata.get('doc_type') == 'نظام':
                result['final_score'] *= 1.2
            
            # Boost relevant categories
            relevant_categories = ['استثمار', 'قانون الشركات', 'تسجيل تجاري']
            if metadata.get('category') in relevant_categories:
                result['final_score'] *= 1.2
        
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
    
    def _rerank_for_procedures(self, results: List[Dict]) -> List[Dict]:
        """Rerank results for procedural queries"""
        for result in results:
            metadata = result['metadata']
            
            # Boost instructions and systems for procedures
            if metadata.get('doc_type') == 'تعليمات':
                result['final_score'] *= 1.3
            elif metadata.get('doc_type') == 'نظام':
                result['final_score'] *= 1.2
            
            # Boost article-level chunks for precise procedures
            if metadata.get('chunk_type') == 'article':
                result['final_score'] *= 1.1
        
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
    
    def _procedure_reasoning(self, query: str, docs: List[Dict], history: List[str]) -> QueryResult:
        """Reasoning for procedural queries"""
        reasoning_steps = [
            "تحديد الإجراءات القانونية المطلوبة",
            "ترتيب الخطوات حسب التسلسل القانوني",
            "تحديد الجهات المختصة لكل خطوة",
            "استخراج المتطلبات والوثائق اللازمة"
        ]
        
        return self._generate_structured_answer(query, docs, reasoning_steps, "procedural", history)
    
    def _requirements_reasoning(self, query: str, docs: List[Dict], history: List[str]) -> QueryResult:
        """Reasoning for requirements queries"""
        reasoning_steps = [
            "تحديد الشروط الأساسية من القوانين",
            "استخراج المتطلبات التفصيلية من الأنظمة",
            "تجميع التعليمات التنفيذية",
            "ترتيب الشروط حسب الأولوية"
        ]
        
        return self._generate_structured_answer(query, docs, reasoning_steps, "requirements", history)
    
    def _establishment_reasoning(self, query: str, docs: List[Dict], history: List[str]) -> QueryResult:
        """Specialized reasoning for business establishment queries"""
        reasoning_steps = [
            "تحديد نوع المنشأة المطلوب تأسيسها",
            "استخراج الشروط القانونية للتأسيس",
            "تحديد الإجراءات الحكومية المطلوبة", 
            "ترتيب خطوات التأسيس زمنياً",
            "تحديد الجهات والوزارات المختصة"
        ]
        
        return self._generate_structured_answer(query, docs, reasoning_steps, "establishment", history)
    
    def _registration_reasoning(self, query: str, docs: List[Dict], history: List[str]) -> QueryResult:
        """Reasoning for registration queries"""
        reasoning_steps = [
            "تحديد نوع التسجيل المطلوب",
            "استخراج شروط التسجيل",
            "تحديد الوثائق المطلوبة",
            "تحديد الرسوم والمدد الزمنية"
        ]
        
        return self._generate_structured_answer(query, docs, reasoning_steps, "registration", history)
    
    def _general_reasoning(self, query: str, docs: List[Dict], history: List[str]) -> QueryResult:
        """General reasoning for other queries"""
        reasoning_steps = [
            "تحليل السؤال القانوني",
            "البحث في النصوص القانونية ذات الصلة",
            "استخراج المعلومات المطلوبة",
            "تنظيم الإجابة وفق التسلسل القانوني"
        ]
        
        return self._generate_structured_answer(query, docs, reasoning_steps, "general", history)
    
    def _generate_structured_answer(self, query: str, docs: List[Dict], reasoning_steps: List[str], answer_type: str, conversation_history: List[str] = None) -> QueryResult:
        """Generate structured legal answer using OpenAI"""
        from openai import OpenAI
        client = OpenAI()
        
        # Detect query language
        detected_language = self.detect_language(query)
        
        # Prepare conversation context if available - increased for more comprehensive context
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            conversation_context = f"\n\nسياق المحادثة السابقة:\n" + "\n".join(conversation_history[-8:])  # Last 8 messages for more comprehensive context
        
        # Check if this is a follow-up question
        follow_up_keywords = ['هذا', 'ذلك', 'المذكور', 'السابق', 'النقطة', 'البند', 'الخطوة', 'وضح', 'أكثر', 'تفصيل']
        is_follow_up = any(keyword in query.lower() for keyword in follow_up_keywords)
        
        # Prepare context from retrieved documents
        context_parts = []
        sources = []
        citations = []
        
        for i, doc in enumerate(docs[:10], 1):  # Use top 10 documents for comprehensive coverage
            metadata = doc['metadata']
            content = doc['content'][:3000]  # Increased content length for more comprehensive context
            
            # Build proper legal document reference instead of generic "الوثيقة 1"
            if metadata.get('law_number') and metadata.get('year'):
                doc_reference = f"{metadata.get('doc_type', 'قانون')} رقم {metadata['law_number']} لسنة {metadata['year']}"
            else:
                doc_reference = f"{metadata.get('doc_type', 'قانون')}: {metadata.get('doc_title', 'غير محدد')}"
            
            # Format context with proper legal reference
            context_parts.append(f"""المرجع القانوني: {doc_reference}
النوع: {metadata.get('doc_type', 'غير محدد')}
المحتوى: {content}""")
            
            sources.append({
                'title': metadata.get('doc_title', 'غير محدد'),
                'type': metadata.get('doc_type', 'غير محدد'),
                'law_number': metadata.get('law_number'),
                'year': metadata.get('year'),
                'relevance_score': doc.get('final_score', 0)
            })
            
            if metadata.get('law_number') and metadata.get('year'):
                citations.append(f"{metadata['doc_type']} رقم {metadata['law_number']} لسنة {metadata['year']}")
        
        context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        
        # Create language-specific prompts
        if detected_language == 'english':
            # English prompts
            if answer_type == "establishment":
                prompt = f"""You are a legal expert specialized in Jordanian laws. Answer the following question based on the attached legal documents.

Jordanian Legal Documents:
{context}{conversation_context}

Question: {query}

Important formatting and content instructions (Provide the most comprehensive and detailed answer possible):
1. Start with a comprehensive title that summarizes the topic followed by a colon
2. Organize the answer in detailed and extensive numbered points, each point containing:
   - A comprehensive descriptive title followed by a colon (without any formatting symbols)
   - Multiple detailed sub-points starting with dash (-) covering all aspects
   - Specific legal reference for each sub-point
   - Additional details and explanations for each point
3. Use this format for references: "Legal Reference: [Law name and number for year], Article [article number]"
4. Do not use any formatting symbols like ** or * or ##
5. Use extensive direct quotes from the legal texts attached above
6. Specify in detail: competent government agencies, exact timeframes, all fees, required documents, conditions
7. Arrange steps chronologically for implementation with detailed explanation of each step
8. Include all exceptions, special cases, and additional regulations
9. Use only information found in the documents attached above
10. Do not use generic references like "Document 1" or "Document 2"
11. Provide comprehensive coverage of all aspects related to the topic
12. Include any supplementary information or helpful details from the documents
13. Explain the legal background and objectives of the legislation
14. End with: "Notice: This information is for general guidance only. Consult a qualified attorney for personal legal advice."

Answer:"""
            
            elif answer_type == "procedural":
                prompt = f"""You are a legal expert specialized in Jordanian laws. Explain the legal procedures based on the attached documents.

Jordanian Legal Documents:
{context}{conversation_context}

Question: {query}

Important formatting and content instructions:
1. Start with a title that summarizes the procedures followed by a colon
2. Organize the answer in numbered points, each point containing:
   - A descriptive title for each procedure followed by a colon (without any formatting symbols)
   - Sub-points starting with dash (-) explaining procedure details
   - Specific legal reference for each sub-point
3. Use this format for references: "Legal Reference: [Law name and number for year], Article [article number]"
4. Do not use any formatting symbols like ** or * or ##
5. Use direct quotes from the legal texts attached above
6. Specify competent agency, required documents, and timeframes
7. Clarify any special conditions or controls
8. Use only information from the documents attached above
9. Do not use generic references like "Document 1" or "Document 2"
10. End with: "Notice: This information is for general guidance only. Consult a qualified attorney for personal legal advice."

Required Procedures:"""
            
            else:  # general, requirements, registration
                prompt = f"""You are a legal expert specialized in Jordanian laws. Answer the question based on the attached legal documents.

Jordanian Legal Documents:
{context}{conversation_context}

Question: {query}

Important formatting and content instructions:
1. Start with a title that summarizes the topic followed by a colon
2. Organize the answer in numbered points, each point containing:
   - A descriptive title followed by a colon (without any formatting symbols)
   - Sub-points starting with dash (-) explaining the details
   - Specific legal reference for each sub-point
3. Use this format for references: "Legal Reference: [Law name and number for year], Article [article number]"
4. Do not use any formatting symbols like ** or * or ##
5. Use direct quotes from the legal texts attached above in format: "Quote: [literal text]"
6. Clarify any exceptions or special conditions
7. Use only information found in the documents attached above
8. Do not use generic references like "Document 1" or "Document 2"
9. If specific information is not found, mention that the information needs confirmation from an attorney
10. End with: "Notice: This information is for general guidance only. Consult a qualified attorney for personal legal advice."

Answer:"""
        
        else:  # Arabic prompts (default)
            if answer_type == "establishment":
                prompt = f"""أنت خبير قانوني متخصص في القوانين الأردنية. قم بالإجابة على السؤال التالي بناءً على الوثائق القانونية المرفقة.

الوثائق القانونية الأردنية:
{context}{conversation_context}

السؤال: {query}

تعليمات مهمة للتنسيق والمحتوى (أجب بأكبر قدر من التفاصيل والشمولية):
1. ابدأ بعنوان يلخص الموضوع متبوعاً بنقطتين
2. نظم الإجابة بصيغة نقاط مرقمة شاملة ومفصلة، كل نقطة تحتوي على:
   - عنوان وصفي مفصل متبوع بنقطتين (بدون أي رموز تنسيق)
   - نقاط فرعية متعددة وشاملة تبدأ بشرطة (-) تغطي جميع الجوانب
   - مرجع قانوني محدد لكل نقطة فرعية
   - تفاصيل إضافية عن كل نقطة
3. استخدم الصيغة التالية للمراجع: "المرجع القانوني: [اسم القانون ورقمه ولسنته]، المادة [رقم المادة]"
4. لا تستخدم أي رموز تنسيق مثل ** أو * أو ##
5. استخدم اقتباسات مباشرة ومطولة من النصوص القانونية المرفقة أعلاه
6. حدد بتفصيل شامل: الجهات الحكومية المختصة، المدد الزمنية، الرسوم، الوثائق المطلوبة، الشروط الواجب توفرها
7. رتب الخطوات حسب التسلسل الزمني للتنفيذ مع شرح مفصل لكل خطوة
8. اذكر جميع الاستثناءات والحالات الخاصة والضوابط الإضافية
9. استخدم فقط المعلومات الموجودة في الوثائق المرفقة أعلاه
10. لا تستخدم مراجع عامة مثل "الوثيقة 1" أو "الوثيقة 2"
11. قدم شرحاً شاملاً ومفصلاً يغطي جميع الجوانب المتعلقة بالموضوع
12. اذكر أي تفاصيل إضافية أو معلومات مساعدة موجودة في الوثائق
13. اختتم بـ: "تنبيه: هذه المعلومات للإرشاد العام فقط. استشر محامياً مؤهلاً للمشورة القانونية الشخصية."

الإجابة:"""
            
            elif answer_type == "procedural":
                prompt = f"""أنت خبير قانوني متخصص في القوانين الأردنية. قم بشرح الإجراءات القانونية بناءً على الوثائق المرفقة.

الوثائق القانونية الأردنية:
{context}{conversation_context}

السؤال: {query}

تعليمات مهمة للتنسيق والمحتوى (قدم أشمل وأدق التفاصيل الممكنة):
1. ابدأ بعنوان شامل يلخص جميع الإجراءات متبوعاً بنقطتين
2. نظم الإجابة بصيغة نقاط مرقمة مفصلة وشاملة، كل نقطة تحتوي على:
   - عنوان وصفي مفصل لكل إجراء متبوع بنقطتين (بدون أي رموز تنسيق)
   - نقاط فرعية متعددة ومفصلة تبدأ بشرطة (-) تشرح جميع تفاصيل الإجراء
   - مرجع قانوني محدد لكل نقطة فرعية
   - خطوات فرعية تحت كل إجراء رئيسي
3. استخدم الصيغة التالية للمراجع: "المرجع القانوني: [اسم القانون ورقمه ولسنته]، المادة [رقم المادة]"
4. لا تستخدم أي رموز تنسيق مثل ** أو * أو ##
5. استخدم اقتباسات مباشرة ومطولة من النصوص القانونية المرفقة أعلاه
6. حدد بتفصيل كامل: الجهة المختصة، جميع الوثائق المطلوبة، المدد الزمنية الدقيقة، الرسوم المالية
7. وضح جميع الشروط والضوابط الخاصة والاستثناءات
8. اذكر التفاصيل الإجرائية الدقيقة وأي متطلبات إضافية
9. استخدم فقط المعلومات من الوثائق المرفقة أعلاه
10. لا تستخدم مراجع عامة مثل "الوثيقة 1" أو "الوثيقة 2"
11. اذكر أي بدائل إجرائية أو طرق متعددة للتنفيذ
12. قدم معلومات شاملة عن كل خطوة من خطوات العملية
13. اختتم بـ: "تنبيه: هذه المعلومات للإرشاد العام فقط. استشر محامياً مؤهلاً للمشورة القانونية الشخصية."

الإجراءات المطلوبة:"""
            
            else:  # general, requirements, registration
                prompt = f"""أنت خبير قانوني متخصص في القوانين الأردنية. أجب على السؤال بناءً على الوثائق القانونية المرفقة.

الوثائق القانونية الأردنية:
{context}{conversation_context}

السؤال: {query}

تعليمات مهمة للتنسيق والمحتوى (قدم أشمل إجابة ممكنة مع جميع التفاصيل):
1. ابدأ بعنوان شامل يلخص الموضوع متبوعاً بنقطتين
2. نظم الإجابة بصيغة نقاط مرقمة مفصلة جداً وشاملة، كل نقطة تحتوي على:
   - عنوان وصفي مفصل متبوع بنقطتين (بدون أي رموز تنسيق)
   - نقاط فرعية متعددة ومفصلة تبدأ بشرطة (-) تشرح جميع التفاصيل والجوانب
   - مرجع قانوني محدد لكل نقطة فرعية
   - توضيحات إضافية وأمثلة عملية
3. استخدم الصيغة التالية للمراجع: "المرجع القانوني: [اسم القانون ورقمه ولسنته]، المادة [رقم المادة]"
4. لا تستخدم أي رموز تنسيق مثل ** أو * أو ##
5. استخدم اقتباسات مباشرة ومطولة من النصوص القانونية المرفقة أعلاه بالصيغة: "اقتباس: [النص الحرفي]"
6. وضح جميع الاستثناءات والشروط الخاصة والحالات المختلفة
7. اذكر جميع التفاصيل الفنية والإجرائية والعملية
8. اشرح الخلفية القانونية والأهداف من التشريع
9. استخدم فقط المعلومات الموجودة في الوثائق المرفقة أعلاه
10. لا تستخدم مراجع عامة مثل "الوثيقة 1" أو "الوثيقة 2"
11. إذا لم تجد معلومة محددة، اذكر أن المعلومة تحتاج تأكيد من محام
12. قدم معلومات مساعدة وإرشادات عملية
13. اذكر أي تطبيقات عملية أو حالات خاصة
14. اختتم بـ: "تنبيه: هذه المعلومات للإرشاد العام فقط. استشر محامياً مؤهلاً للمشورة القانونية الشخصية."

الإجابة:"""
        
        # Generate answer
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4000  # Increased for more comprehensive responses
            )
            
            answer = response.choices[0].message.content
            confidence = 0.85  # High confidence with advanced system
            
        except Exception as e:
            if detected_language == 'english':
                answer = f"Sorry, an error occurred while processing the question: {str(e)}\n\nPlease try again or consult a specialized attorney."
            else:
                answer = f"عذراً، حدث خطأ في معالجة السؤال: {str(e)}\n\nيرجى إعادة المحاولة أو استشارة محام مختص."
            confidence = 0.0
        
        return QueryResult(
            answer=self.clean_response_formatting(answer),
            sources=sources,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            citations=citations,
            query_type=answer_type,
            processing_time=0.0  # Will be set by calling function
        )

class AdvancedLegalRAGSystem:
    """Main orchestrator for the advanced legal RAG system"""
    
    def __init__(self, documents_path: str = "mit_jordan_data/txt_output"):
        self.documents_path = Path(documents_path)
        self.processor = AdvancedLegalProcessor()
        self.vector_store = AdvancedVectorStore()
        self.reasoning_engine = LegalReasoningEngine(self.vector_store)
        self.documents: List[LegalDocument] = []
        
        print("🚀 Advanced Legal RAG System initialized")
    
    def load_documents(self, force_rebuild: bool = False) -> int:
        """Load documents using pre-built embeddings"""
        print(f"📁 Loading documents using pre-built embeddings...")
        
        # Load pre-built embeddings directly
        if self.vector_store.load_prebuilt_embeddings():
            # Create placeholder documents for compatibility with existing code
            # The actual documents are already loaded in the vector store
            collection_count = self.vector_store.collection.count()
            print(f"📚 Successfully loaded {collection_count} document chunks from pre-built embeddings")
            
            # Create a minimal document list for compatibility
            self.documents = [LegalDocument(
                id="prebuilt_docs",
                title="Pre-built Legal Documents",
                source_file="embeddings_data.json",
                content="Loaded from pre-built embeddings",
                doc_type="متنوعة",
                category="عام",
                law_number=None,
                year=None,
                articles=[],
                metadata={"loaded_from": "pre-built embeddings"},
                word_count=0
            )]
            
            return collection_count
        else:
            print(f"❌ Could not load pre-built embeddings")
            return 0
    
    def query(self, query: str, conversation_history: List[str] = None) -> QueryResult:
        """Process a legal query and return comprehensive result"""
        if not self.documents:
            return QueryResult(
                answer="عذراً، لم يتم تحميل أي وثائق قانونية بعد. يرجى تحميل الوثائق أولاً.",
                sources=[],
                confidence=0.0,
                reasoning_steps=[],
                citations=[],
                query_type="error",
                processing_time=0.0
            )
        
        print(f"🔍 Processing query: {query}")
        return self.reasoning_engine.process_query(query, conversation_history)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        if not self.documents:
            return {"status": "No documents loaded"}
        
        doc_types = {}
        categories = {}
        total_articles = 0
        
        for doc in self.documents:
            # Count document types
            doc_type = doc.doc_type
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            # Count categories
            category = doc.category
            categories[category] = categories.get(category, 0) + 1
            
            # Count articles
            total_articles += len(doc.articles)
        
        return {
            "total_documents": len(self.documents),
            "total_articles": total_articles,
            "document_types": doc_types,
            "legal_categories": categories,
            "vector_store_status": "Ready" if self.vector_store.tfidf_matrix is not None else "Not Ready",
            "average_articles_per_doc": total_articles / len(self.documents) if self.documents else 0
        }

# Main execution
if __name__ == "__main__":
    # Initialize the advanced system
    system = AdvancedLegalRAGSystem()
    
    # Load all documents
    num_docs = system.load_documents()
    
    if num_docs > 0:
        print(f"\n🎉 Advanced Legal RAG System ready with {num_docs} documents!")
        print("\n📊 System Statistics:")
        stats = system.get_system_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n🚀 Ready to process legal queries!")
        print("Use: result = system.query('your question here')")
    else:
        print("❌ No documents loaded. Please check the documents path.") 