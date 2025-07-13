from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
import nltk
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentSummarizer:
    """Handles document summarization using various algorithms"""
    
    def __init__(self, algorithm='textrank'):
        """
        Initialize summarizer
        
        Args:
            algorithm: Summarization algorithm ('textrank', 'lsa', 'luhn')
        """
        self.algorithm = algorithm
        self.target_word_count = 150
        
        # Initialize summarizer based on algorithm
        if algorithm == 'textrank':
            self.summarizer = TextRankSummarizer()
        elif algorithm == 'lsa':
            self.summarizer = LsaSummarizer()
        elif algorithm == 'luhn':
            self.summarizer = LuhnSummarizer()
        else:
            self.summarizer = TextRankSummarizer()
    
    def summarize(self, text: str, max_words: int = 150) -> str:
        """
        Generate summary of the document
        
        Args:
            text: Input text to summarize
            max_words: Maximum words in summary
            
        Returns:
            str: Generated summary
        """
        try:
            if not text or not text.strip():
                return "No content available for summarization."
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            if len(cleaned_text.split()) < 10:
                return "Document too short for meaningful summarization."
            
            # Parse the text
            parser = PlaintextParser.from_string(cleaned_text, Tokenizer("english"))
            
            # Calculate optimal sentence count based on text length
            sentence_count = self._calculate_sentence_count(cleaned_text, max_words)
            
            # Generate summary
            summary_sentences = self.summarizer(parser.document, sentence_count)
            
            # Convert to string
            summary_text = " ".join(str(sentence) for sentence in summary_sentences)
            
            # Ensure summary doesn't exceed word limit
            summary_text = self._trim_to_word_limit(summary_text, max_words)
            
            if not summary_text.strip():
                return "Unable to generate meaningful summary from the provided text."
            
            return summary_text
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for summarization
        
        Args:
            text: Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', ' ', text)
        
        # Fix sentence endings
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\!+', '!', text)
        text = re.sub(r'\?+', '?', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        
        return text.strip()
    
    def _calculate_sentence_count(self, text: str, max_words: int) -> int:
        """
        Calculate optimal number of sentences for summary
        
        Args:
            text: Input text
            max_words: Maximum words allowed
            
        Returns:
            int: Number of sentences to extract
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        total_sentences = len(sentences)
        
        if total_sentences == 0:
            return 1
        
        # Calculate average words per sentence
        total_words = len(text.split())
        avg_words_per_sentence = total_words / total_sentences
        
        # Calculate target sentence count
        target_sentences = max(1, int(max_words / avg_words_per_sentence))
        
        # Ensure we don't exceed available sentences
        target_sentences = min(target_sentences, total_sentences)
        
        # Ensure minimum of 1 and maximum of 10 sentences
        target_sentences = max(1, min(10, target_sentences))
        
        return target_sentences
    
    def _trim_to_word_limit(self, text: str, max_words: int) -> str:
        """
        Trim text to specified word limit
        
        Args:
            text: Input text
            max_words: Maximum words allowed
            
        Returns:
            str: Trimmed text
        """
        words = text.split()
        
        if len(words) <= max_words:
            return text
        
        # Trim to word limit
        trimmed_words = words[:max_words]
        trimmed_text = " ".join(trimmed_words)
        
        # Try to end at a sentence boundary
        last_sentence_end = max(
            trimmed_text.rfind('.'),
            trimmed_text.rfind('!'),
            trimmed_text.rfind('?')
        )
        
        if last_sentence_end > len(trimmed_text) * 0.7:  # If we can keep 70% of content
            trimmed_text = trimmed_text[:last_sentence_end + 1]
        else:
            # Add ellipsis if we had to cut mid-sentence
            trimmed_text += "..."
        
        return trimmed_text
    
    def get_summary_stats(self, original_text: str, summary: str) -> dict:
        """
        Get statistics comparing original text and summary
        
        Args:
            original_text: Original document text
            summary: Generated summary
            
        Returns:
            dict: Comparison statistics
        """
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        
        compression_ratio = (summary_words / original_words) * 100 if original_words > 0 else 0
        
        return {
            'original_words': original_words,
            'summary_words': summary_words,
            'compression_ratio': round(compression_ratio, 2),
            'reduction_percentage': round(100 - compression_ratio, 2)
        }
    
    def generate_multiple_summaries(self, text: str, max_words: int = 150) -> dict:
        """
        Generate summaries using different algorithms
        
        Args:
            text: Input text
            max_words: Maximum words per summary
            
        Returns:
            dict: Summaries from different algorithms
        """
        algorithms = ['textrank', 'lsa', 'luhn']
        summaries = {}
        
        for algo in algorithms:
            try:
                summarizer = DocumentSummarizer(algorithm=algo)
                summary = summarizer.summarize(text, max_words)
                summaries[algo] = summary
            except Exception as e:
                summaries[algo] = f"Error with {algo}: {str(e)}"
        
        return summaries