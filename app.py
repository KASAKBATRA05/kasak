import streamlit as st
import os
from backend.reader import DocumentReader
from backend.summarizer import DocumentSummarizer
from backend.qna_engine import QnAEngine
from backend.challenge_mode import ChallengeMode
from utils.embeddings import EmbeddingManager

# Page config
st.set_page_config(
    page_title="GenAI NLP Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'document_text' not in st.session_state:
        st.session_state.document_text = None
    if 'document_summary' not in st.session_state:
        st.session_state.document_summary = None
    if 'qna_engine' not in st.session_state:
        st.session_state.qna_engine = None
    if 'challenge_questions' not in st.session_state:
        st.session_state.challenge_questions = None
    if 'current_question_idx' not in st.session_state:
        st.session_state.current_question_idx = 0
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = []

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 GenAI NLP Document Assistant</h1>
        <p>Upload documents, get summaries, ask questions, and test your knowledge!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        mode = st.selectbox(
            "Choose Mode:",
            ["📄 Document Upload", "📝 Summary", "❓ Ask Questions", "🎯 Challenge Mode"]
        )
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This assistant uses:
        - **PyMuPDF** for PDF reading
        - **Sumy TextRank** for summarization
        - **Sentence Transformers** for embeddings
        - **RoBERTa** for question answering
        - **FAISS** for similarity search
        """)
    
    # Main content based on selected mode
    if mode == "📄 Document Upload":
        document_upload_page()
    elif mode == "📝 Summary":
        summary_page()
    elif mode == "❓ Ask Questions":
        qna_page()
    elif mode == "🎯 Challenge Mode":
        challenge_page()

def document_upload_page():
    st.header("📄 Document Upload")
    
    st.markdown("""
    <div class="feature-card">
        <h3>Upload Your Document</h3>
        <p>Supported formats: PDF, TXT files</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        help="Upload a PDF or TXT file to get started"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            try:
                # Initialize document reader
                reader = DocumentReader()
                
                # Extract text based on file type
                if uploaded_file.type == "application/pdf":
                    text = reader.extract_from_pdf(uploaded_file)
                else:
                    text = reader.extract_from_txt(uploaded_file)
                
                # Store in session state
                st.session_state.document_text = text
                
                # Display success message
                st.markdown("""
                <div class="success-box">
                    ✅ Document processed successfully!
                </div>
                """, unsafe_allow_html=True)
                
                # Show document preview
                st.subheader("📖 Document Preview")
                preview_text = text[:1000] + "..." if len(text) > 1000 else text
                st.text_area("Content Preview", preview_text, height=200, disabled=True)
                
                # Document stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Characters", len(text))
                with col2:
                    st.metric("Words", len(text.split()))
                with col3:
                    st.metric("Paragraphs", len(text.split('\n\n')))
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    ❌ Error processing document: {str(e)}
                </div>
                """, unsafe_allow_html=True)

def summary_page():
    st.header("📝 Document Summary")
    
    if st.session_state.document_text is None:
        st.markdown("""
        <div class="info-box">
            ℹ️ Please upload a document first in the Document Upload section.
        </div>
        """, unsafe_allow_html=True)
        return
    
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating summary..."):
            try:
                summarizer = DocumentSummarizer()
                summary = summarizer.summarize(st.session_state.document_text)
                st.session_state.document_summary = summary
                
                st.markdown("""
                <div class="success-box">
                    ✅ Summary generated successfully!
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    ❌ Error generating summary: {str(e)}
                </div>
                """, unsafe_allow_html=True)
    
    if st.session_state.document_summary:
        st.subheader("📋 Summary")
        st.markdown(f"""
        <div class="feature-card">
            <p>{st.session_state.document_summary}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary stats
        summary_words = len(st.session_state.document_summary.split())
        st.metric("Summary Word Count", summary_words)

def qna_page():
    st.header("❓ Ask Questions")
    
    if st.session_state.document_text is None:
        st.markdown("""
        <div class="info-box">
            ℹ️ Please upload a document first in the Document Upload section.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize QnA engine if not already done
    if st.session_state.qna_engine is None:
        with st.spinner("Initializing Q&A system..."):
            try:
                st.session_state.qna_engine = QnAEngine()
                st.session_state.qna_engine.index_document(st.session_state.document_text)
                
                st.markdown("""
                <div class="success-box">
                    ✅ Q&A system ready!
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    ❌ Error initializing Q&A system: {str(e)}
                </div>
                """, unsafe_allow_html=True)
                return
    
    # Question input
    user_question = st.text_input(
        "Ask a question about the document:",
        placeholder="e.g., What is the main topic discussed?"
    )
    
    if st.button("Get Answer", type="primary") and user_question:
        with st.spinner("Finding answer..."):
            try:
                result = st.session_state.qna_engine.answer_question(user_question)
                
                st.subheader("💡 Answer")
                st.markdown(f"""
                <div class="feature-card">
                    <p><strong>Answer:</strong> {result['answer']}</p>
                    <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if result['context']:
                    st.subheader("📖 Relevant Context")
                    st.text_area("Context", result['context'], height=150, disabled=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    ❌ Error finding answer: {str(e)}
                </div>
                """, unsafe_allow_html=True)

def challenge_page():
    st.header("🎯 Challenge Mode")
    
    if st.session_state.document_text is None:
        st.markdown("""
        <div class="info-box">
            ℹ️ Please upload a document first in the Document Upload section.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Generate questions if not already done
    if st.session_state.challenge_questions is None:
        if st.button("Generate Challenge Questions", type="primary"):
            with st.spinner("Generating questions..."):
                try:
                    challenge_mode = ChallengeMode()
                    questions = challenge_mode.generate_questions(st.session_state.document_text)
                    st.session_state.challenge_questions = questions
                    st.session_state.current_question_idx = 0
                    st.session_state.user_answers = []
                    
                    st.markdown("""
                    <div class="success-box">
                        ✅ Challenge questions generated!
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-box">
                        ❌ Error generating questions: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Display questions and handle answers
    if st.session_state.challenge_questions:
        questions = st.session_state.challenge_questions
        current_idx = st.session_state.current_question_idx
        
        if current_idx < len(questions):
            current_q = questions[current_idx]
            
            st.subheader(f"Question {current_idx + 1} of {len(questions)}")
            st.markdown(f"""
            <div class="feature-card">
                <h4>{current_q['question']}</h4>
                <p><strong>Type:</strong> {current_q['type']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Answer input based on question type
            if current_q['type'] == 'yes_no':
                user_answer = st.radio("Your answer:", ["Yes", "No"])
            else:
                user_answer = st.text_input("Your answer:")
            
            if st.button("Submit Answer", type="primary") and user_answer:
                # Evaluate answer
                challenge_mode = ChallengeMode()
                evaluation = challenge_mode.evaluate_answer(
                    user_answer, 
                    current_q['correct_answer'],
                    current_q['context']
                )
                
                # Store answer
                st.session_state.user_answers.append({
                    'question': current_q['question'],
                    'user_answer': user_answer,
                    'correct_answer': current_q['correct_answer'],
                    'evaluation': evaluation
                })
                
                # Show evaluation
                if evaluation['is_correct']:
                    st.markdown("""
                    <div class="success-box">
                        ✅ Correct! Well done!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        ❌ Incorrect. The correct answer is: {current_q['correct_answer']}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>Explanation:</strong> {evaluation['explanation']}
                </div>
                """, unsafe_allow_html=True)
                
                # Move to next question
                st.session_state.current_question_idx += 1
                
                if st.button("Next Question"):
                    st.rerun()
        
        else:
            # Show final results
            st.subheader("🏆 Challenge Complete!")
            
            correct_answers = sum(1 for ans in st.session_state.user_answers if ans['evaluation']['is_correct'])
            total_questions = len(st.session_state.user_answers)
            score = (correct_answers / total_questions) * 100
            
            st.markdown(f"""
            <div class="feature-card">
                <h3>Your Score: {correct_answers}/{total_questions} ({score:.1f}%)</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Show detailed results
            for i, answer in enumerate(st.session_state.user_answers):
                status = "✅" if answer['evaluation']['is_correct'] else "❌"
                st.markdown(f"""
                <div class="feature-card">
                    <p><strong>Q{i+1}:</strong> {answer['question']}</p>
                    <p><strong>Your Answer:</strong> {answer['user_answer']} {status}</p>
                    <p><strong>Correct Answer:</strong> {answer['correct_answer']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("Start New Challenge"):
                st.session_state.challenge_questions = None
                st.session_state.current_question_idx = 0
                st.session_state.user_answers = []
                st.rerun()

if __name__ == "__main__":
    main()