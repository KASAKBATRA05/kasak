from haystack.nodes import FARMReader, EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import Document
import re
from typing import Dict, List, Any

class QnAEngine:
    """Question-Answering engine using Haystack with RoBERTa and Sentence Transformers"""
    
    def __init__(self):
        """Initialize the Q&A engine components"""
        try:
            # Initialize document store
            self.document_store = InMemoryDocumentStore()
            
            # Initialize retriever with sentence transformers
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                use_gpu=False
            )
            
            # Initialize reader with RoBERTa
            self.reader = FARMReader(
                model_name_or_path="deepset/roberta-base-squad2",
                use_gpu=False,
                top_k=1,
                return_no_answer=True
            )
            
            # Create pipeline
            self.pipeline = ExtractiveQAPipeline(self.reader, self.retriever)
            
            self.is_indexed = False
            
        except Exception as e:
            raise Exception(f"Error initializing Q&A engine: {str(e)}")
    
    def index_document(self, text: str) -> None:
        """
        Index the document for question answering
        
        Args:
            text: Document text to index
        """
        try:
            if not text or not text.strip():
                raise ValueError("No text provided for indexing")
            
            # Split text into paragraphs for better retrieval
            paragraphs = self._split_into_paragraphs(text)
            
            if not paragraphs:
                raise ValueError("No valid paragraphs found in text")
            
            # Create Haystack documents
            documents = []
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():  # Only add non-empty paragraphs
                    doc = Document(
                        content=paragraph,
                        meta={"paragraph_id": i, "source": "uploaded_document"}
                    )
                    documents.append(doc)
            
            if not documents:
                raise ValueError("No valid documents created from text")
            
            # Clear existing documents
            self.document_store.delete_documents()
            
            # Write documents to store
            self.document_store.write_documents(documents)
            
            # Update embeddings
            self.document_store.update_embeddings(self.retriever)
            
            self.is_indexed = True
            
        except Exception as e:
            raise Exception(f"Error indexing document: {str(e)}")
    
    def answer_question(self, question: str, top_k_retriever: int = 5, top_k_reader: int = 1) -> Dict[str, Any]:
        """
        Answer a question based on the indexed document
        
        Args:
            question: User's question
            top_k_retriever: Number of documents to retrieve
            top_k_reader: Number of answers to return
            
        Returns:
            dict: Answer with confidence and context
        """
        try:
            if not self.is_indexed:
                raise ValueError("No document has been indexed yet")
            
            if not question or not question.strip():
                raise ValueError("No question provided")
            
            # Clean the question
            question = self._clean_question(question)
            
            # Run the pipeline
            result = self.pipeline.run(
                query=question,
                params={
                    "Retriever": {"top_k": top_k_retriever},
                    "Reader": {"top_k": top_k_reader}
                }
            )
            
            # Extract answer information
            if result['answers'] and len(result['answers']) > 0:
                best_answer = result['answers'][0]
                
                return {
                    'answer': best_answer.answer,
                    'confidence': best_answer.score,
                    'context': best_answer.context,
                    'start_idx': best_answer.offsets_in_context[0].start if best_answer.offsets_in_context else None,
                    'end_idx': best_answer.offsets_in_context[0].end if best_answer.offsets_in_context else None,
                    'document_id': best_answer.document_ids[0] if best_answer.document_ids else None
                }
            else:
                return {
                    'answer': "I couldn't find a relevant answer in the document.",
                    'confidence': 0.0,
                    'context': None,
                    'start_idx': None,
                    'end_idx': None,
                    'document_id': None
                }
                
        except Exception as e:
            return {
                'answer': f"Error processing question: {str(e)}",
                'confidence': 0.0,
                'context': None,
                'start_idx': None,
                'end_idx': None,
                'document_id': None
            }
    
    def get_similar_passages(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get similar passages for a question without running the reader
        
        Args:
            question: User's question
            top_k: Number of passages to return
            
        Returns:
            list: Similar passages with scores
        """
        try:
            if not self.is_indexed:
                return []
            
            # Use retriever to find similar passages
            documents = self.retriever.retrieve(query=question, top_k=top_k)
            
            passages = []
            for doc in documents:
                passages.append({
                    'content': doc.content,
                    'score': doc.score,
                    'meta': doc.meta
                })
            
            return passages
            
        except Exception as e:
            return []
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into meaningful paragraphs
        
        Args:
            text: Input text
            
        Returns:
            list: List of paragraphs
        """
        # Split by double newlines first
        paragraphs = text.split('\n\n')
        
        # Further split long paragraphs
        final_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is too long, split by sentences
            if len(para.split()) > 200:  # More than 200 words
                sentences = re.split(r'[.!?]+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # If adding this sentence would make chunk too long, save current chunk
                    if current_chunk and len((current_chunk + " " + sentence).split()) > 150:
                        final_paragraphs.append(current_chunk)
                        current_chunk = sentence
                    else:
                        if current_chunk:
                            current_chunk += ". " + sentence
                        else:
                            current_chunk = sentence
                
                # Add remaining chunk
                if current_chunk:
                    final_paragraphs.append(current_chunk)
            else:
                final_paragraphs.append(para)
        
        # Filter out very short paragraphs
        final_paragraphs = [p for p in final_paragraphs if len(p.split()) >= 5]
        
        return final_paragraphs
    
    def _clean_question(self, question: str) -> str:
        """
        Clean and normalize the question
        
        Args:
            question: Raw question
            
        Returns:
            str: Cleaned question
        """
        # Remove excessive whitespace
        question = re.sub(r'\s+', ' ', question).strip()
        
        # Ensure question ends with question mark if it's a question
        question_words = ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose']
        if any(question.lower().startswith(word) for word in question_words):
            if not question.endswith('?'):
                question += '?'
        
        return question
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed document
        
        Returns:
            dict: Document statistics
        """
        if not self.is_indexed:
            return {'error': 'No document indexed'}
        
        try:
            # Get all documents
            all_docs = self.document_store.get_all_documents()
            
            total_docs = len(all_docs)
            total_chars = sum(len(doc.content) for doc in all_docs)
            total_words = sum(len(doc.content.split()) for doc in all_docs)
            
            return {
                'total_paragraphs': total_docs,
                'total_characters': total_chars,
                'total_words': total_words,
                'average_paragraph_length': total_words / total_docs if total_docs > 0 else 0
            }
            
        except Exception as e:
            return {'error': f'Error getting stats: {str(e)}'}