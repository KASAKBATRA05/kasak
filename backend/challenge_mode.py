import re
import random
from typing import List, Dict, Any
from fuzzywuzzy import fuzz
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ChallengeMode:
    """Generate and evaluate challenge questions from document content"""
    
    def __init__(self):
        """Initialize challenge mode"""
        self.stop_words = set(stopwords.words('english'))
        self.question_types = ['fill_blank', 'yes_no', 'short_answer']
    
    def generate_questions(self, text: str, num_questions: int = 3) -> List[Dict[str, Any]]:
        """
        Generate challenge questions from the document
        
        Args:
            text: Document text
            num_questions: Number of questions to generate
            
        Returns:
            list: Generated questions with answers and context
        """
        try:
            if not text or not text.strip():
                return []
            
            # Extract informative sentences
            sentences = self._extract_informative_sentences(text)
            
            if len(sentences) < num_questions:
                num_questions = len(sentences)
            
            if num_questions == 0:
                return []
            
            # Select diverse sentences
            selected_sentences = self._select_diverse_sentences(sentences, num_questions)
            
            questions = []
            for i, sentence in enumerate(selected_sentences):
                question_type = self.question_types[i % len(self.question_types)]
                question = self._generate_question_from_sentence(sentence, question_type)
                
                if question:
                    questions.append(question)
            
            return questions
            
        except Exception as e:
            return [{'error': f'Error generating questions: {str(e)}'}]
    
    def evaluate_answer(self, user_answer: str, correct_answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluate user's answer against the correct answer
        
        Args:
            user_answer: User's submitted answer
            correct_answer: Correct answer
            context: Context from which the question was derived
            
        Returns:
            dict: Evaluation results with feedback
        """
        try:
            if not user_answer or not user_answer.strip():
                return {
                    'is_correct': False,
                    'similarity_score': 0,
                    'explanation': 'No answer provided.',
                    'feedback': 'Please provide an answer.'
                }
            
            # Clean answers
            user_answer_clean = self._clean_answer(user_answer)
            correct_answer_clean = self._clean_answer(correct_answer)
            
            # Calculate similarity
            similarity_score = self._calculate_similarity(user_answer_clean, correct_answer_clean)
            
            # Determine if correct (threshold: 70%)
            is_correct = similarity_score >= 70
            
            # Generate explanation
            explanation = self._generate_explanation(
                user_answer_clean, 
                correct_answer_clean, 
                context, 
                is_correct, 
                similarity_score
            )
            
            # Generate feedback
            feedback = self._generate_feedback(is_correct, similarity_score)
            
            return {
                'is_correct': is_correct,
                'similarity_score': similarity_score,
                'explanation': explanation,
                'feedback': feedback
            }
            
        except Exception as e:
            return {
                'is_correct': False,
                'similarity_score': 0,
                'explanation': f'Error evaluating answer: {str(e)}',
                'feedback': 'There was an error processing your answer.'
            }
    
    def _extract_informative_sentences(self, text: str) -> List[str]:
        """
        Extract informative sentences suitable for question generation
        
        Args:
            text: Document text
            
        Returns:
            list: Informative sentences
        """
        sentences = sent_tokenize(text)
        informative_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter criteria
            if (len(sentence.split()) >= 8 and  # At least 8 words
                len(sentence.split()) <= 30 and  # At most 30 words
                not sentence.startswith(('Figure', 'Table', 'Chart')) and  # Skip captions
                '.' in sentence and  # Has proper punctuation
                self._has_meaningful_content(sentence)):  # Has meaningful content
                
                informative_sentences.append(sentence)
        
        return informative_sentences
    
    def _has_meaningful_content(self, sentence: str) -> bool:
        """
        Check if sentence has meaningful content for question generation
        
        Args:
            sentence: Input sentence
            
        Returns:
            bool: True if sentence is meaningful
        """
        # Tokenize and tag
        words = word_tokenize(sentence.lower())
        pos_tags = pos_tag(words)
        
        # Count important parts of speech
        nouns = sum(1 for word, pos in pos_tags if pos.startswith('NN'))
        verbs = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
        adjectives = sum(1 for word, pos in pos_tags if pos.startswith('JJ'))
        
        # Must have at least 2 nouns and 1 verb
        return nouns >= 2 and verbs >= 1
    
    def _select_diverse_sentences(self, sentences: List[str], num_questions: int) -> List[str]:
        """
        Select diverse sentences for question generation
        
        Args:
            sentences: Available sentences
            num_questions: Number of questions needed
            
        Returns:
            list: Selected sentences
        """
        if len(sentences) <= num_questions:
            return sentences
        
        # Simple diversity selection - pick sentences from different parts of text
        selected = []
        step = len(sentences) // num_questions
        
        for i in range(num_questions):
            idx = i * step
            if idx < len(sentences):
                selected.append(sentences[idx])
        
        return selected
    
    def _generate_question_from_sentence(self, sentence: str, question_type: str) -> Dict[str, Any]:
        """
        Generate a question from a sentence based on type
        
        Args:
            sentence: Source sentence
            question_type: Type of question to generate
            
        Returns:
            dict: Question with answer and metadata
        """
        try:
            if question_type == 'fill_blank':
                return self._generate_fill_blank(sentence)
            elif question_type == 'yes_no':
                return self._generate_yes_no(sentence)
            elif question_type == 'short_answer':
                return self._generate_short_answer(sentence)
            else:
                return None
                
        except Exception as e:
            return None
    
    def _generate_fill_blank(self, sentence: str) -> Dict[str, Any]:
        """Generate fill-in-the-blank question"""
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        
        # Find important nouns or adjectives to blank out
        candidates = []
        for i, (word, pos) in enumerate(pos_tags):
            if (pos.startswith('NN') or pos.startswith('JJ')) and word.lower() not in self.stop_words:
                candidates.append((i, word))
        
        if not candidates:
            return None
        
        # Select a word to blank out
        idx, target_word = random.choice(candidates)
        
        # Create question by replacing word with blank
        question_words = words.copy()
        question_words[idx] = "______"
        question_text = " ".join(question_words)
        
        return {
            'question': f"Fill in the blank: {question_text}",
            'correct_answer': target_word,
            'type': 'fill_blank',
            'context': sentence,
            'difficulty': 'medium'
        }
    
    def _generate_yes_no(self, sentence: str) -> Dict[str, Any]:
        """Generate yes/no question"""
        # Simple approach: convert statement to question
        question_starters = [
            "Is it true that",
            "Does the text state that",
            "According to the document, is it correct that"
        ]
        
        starter = random.choice(question_starters)
        question_text = f"{starter} {sentence.lower()}?"
        
        # For simplicity, assume the answer is "Yes" since we're using actual content
        return {
            'question': question_text,
            'correct_answer': 'Yes',
            'type': 'yes_no',
            'context': sentence,
            'difficulty': 'easy'
        }
    
    def _generate_short_answer(self, sentence: str) -> Dict[str, Any]:
        """Generate short answer question"""
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        
        # Find key entities (nouns) to ask about
        key_nouns = []
        for word, pos in pos_tags:
            if pos.startswith('NN') and word.lower() not in self.stop_words and len(word) > 3:
                key_nouns.append(word)
        
        if not key_nouns:
            return None
        
        target_noun = random.choice(key_nouns)
        
        # Generate question asking about the noun
        question_templates = [
            f"What is mentioned about {target_noun.lower()} in the text?",
            f"According to the document, what can you tell about {target_noun.lower()}?",
            f"What information is provided regarding {target_noun.lower()}?"
        ]
        
        question_text = random.choice(question_templates)
        
        return {
            'question': question_text,
            'correct_answer': sentence,  # The whole sentence as context
            'type': 'short_answer',
            'context': sentence,
            'difficulty': 'hard'
        }
    
    def _calculate_similarity(self, answer1: str, answer2: str) -> float:
        """
        Calculate similarity between two answers
        
        Args:
            answer1: First answer
            answer2: Second answer
            
        Returns:
            float: Similarity score (0-100)
        """
        # Use multiple similarity metrics
        ratio = fuzz.ratio(answer1, answer2)
        partial_ratio = fuzz.partial_ratio(answer1, answer2)
        token_sort_ratio = fuzz.token_sort_ratio(answer1, answer2)
        token_set_ratio = fuzz.token_set_ratio(answer1, answer2)
        
        # Weighted average
        similarity = (ratio * 0.3 + partial_ratio * 0.2 + 
                     token_sort_ratio * 0.25 + token_set_ratio * 0.25)
        
        return round(similarity, 2)
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and normalize answer text"""
        if not answer:
            return ""
        
        # Convert to lowercase
        answer = answer.lower().strip()
        
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer)
        
        # Remove common punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        
        return answer
    
    def _generate_explanation(self, user_answer: str, correct_answer: str, 
                            context: str, is_correct: bool, similarity: float) -> str:
        """Generate explanation for the answer evaluation"""
        if is_correct:
            return f"Your answer is correct! The text states: '{context}'"
        else:
            return (f"Your answer differs from the expected response. "
                   f"The relevant text states: '{context}'. "
                   f"Your answer had {similarity}% similarity to the expected answer.")
    
    def _generate_feedback(self, is_correct: bool, similarity: float) -> str:
        """Generate feedback based on correctness and similarity"""
        if is_correct:
            return "Excellent! You understood the content well."
        elif similarity >= 50:
            return "Close! You're on the right track but could be more precise."
        elif similarity >= 25:
            return "Partially correct. Review the context for better understanding."
        else:
            return "Not quite right. Please read the relevant section more carefully."