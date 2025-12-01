import streamlit as st
import os
import time
import json
import pandas as pd
import smtplib
from email.message import EmailMessage
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from fpdf import FPDF

# --- 1. home page ---
st.set_page_config(
    page_title="SectorIQ - Enterprise Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. generate PDF ---
def create_pdf(topic, report, quiz, reading_list):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"SectorIQ Report: {topic}", ln=True, align='C')
    pdf.ln(10)
    
    # Report Section
    pdf.set_font("Arial", size=11)
    clean_report = report.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, txt=clean_report)
    pdf.ln(10)
    
    # Reading List Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Curated Reading List", ln=True)
    pdf.set_font("Arial", size=10)
    for item in reading_list:
        title = item['title'].encode('latin-1', 'replace').decode('latin-1')
        url = item['url'].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, txt=f"- {title}\n  {url}")
        pdf.ln(2)
        
    # Quiz Section
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Employee Assessment", ln=True)
    pdf.set_font("Arial", size=11)
    clean_quiz = quiz.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, txt=clean_quiz)
        
    return pdf.output(dest="S").encode("latin-1")

# --- 3. send email ---
def send_email_with_pdf(user_email, email_password, to_email, topic, pdf_bytes):
    msg = EmailMessage()
    msg['Subject'] = f"SectorIQ Training Kit: {topic}"
    msg['From'] = user_email
    msg['To'] = to_email
    msg.set_content(f"""
    Hello,
    
    Here is your AI-generated onboarding package for '{topic}'.
    It includes an executive briefing, a curated reading list, and an assessment quiz.
    
    Powered by SectorIQ Enterprise Agent.
    """)
    
    msg.add_attachment(
        pdf_bytes,
        maintype='application',
        subtype='pdf',
        filename=f"SectorIQ_{topic.replace(' ', '_')}.pdf"
    )
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(user_email, email_password)
            smtp.send_message(msg)
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)

# --- 4. QA check ---
def run_qa_check(llm, report_content, quiz_content):
    qa_prompt = f"""
    You are a Senior Chief Editor. Evaluate the following Industry Report and Quiz.
    Return the result in valid JSON format ONLY.
    Keys: "strategic_depth" (1-10), "clarity" (1-10), "quiz_relevance" (1-10), "overall_score" (1-10), "recommendation" (short text).
    
    Report: {report_content[:2000]}
    Quiz: {quiz_content}
    """
    try:
        response = llm.invoke([HumanMessage(content=qa_prompt)])
        json_str = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except:
        return None

# --- 5. Session State initial ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- 6. sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083256.png", width=50)
    st.title("SectorIQ")
    st.markdown("### ⚙️ System Config")
    
    # AI Keys
    with st.expander("🔑 AI API Keys", expanded=True):
        google_key = st.text_input("Google API Key", type="password", help="Required")
        tavily_key = st.text_input("Tavily API Key", type="password", help="Required")

    # Email Keys (Optional)
    with st.expander("📧 Email Settings (Optional)", expanded=False):
        st.caption("Required to send reports via email.")
        email_user = st.text_input("Gmail Address")
        email_pass = st.text_input("App Password", type="password", help="Use App Password, NOT login password")
    
    st.markdown("---")
    
    # Model Selector
    model_choice = st.selectbox(
        "🧠 AI Model",
        ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        help="Select model capability."
    )
    
    st.markdown("---")
    
    # History
    st.markdown("### 🕒 History")
    if not st.session_state.history:
        st.caption("No reports yet.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"📄 {item['topic']}"):
                st.caption(f"Score: {item.get('qa_score', 'N/A')}/10")
                st.markdown(item['summary'])
                st.download_button(
                    label="📥 PDF",
                    data=create_pdf(item['topic'], item['full_report'], item['quiz'], item['reading_list']),
                    file_name=f"SectorIQ_{item['topic']}.pdf",
                    mime="application/pdf",
                    key=f"hist_btn_{idx}"
                )

# --- 7. Agent Graph build ---

class AgentState(TypedDict):
    topic: str
    raw_content: str
    reading_list: List[dict]
    structured_report: str
    quiz: str

def build_agent_graph(api_key_google, api_key_tavily, model_name):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7, google_api_key=api_key_google)
    tavily = TavilyClient(api_key=api_key_tavily)

    def researcher_agent(state: AgentState):
        try:
            # deep search
            results = tavily.search(
                query=f"{state['topic']} market trends strategic analysis 2025", 
                search_depth="advanced", 
                max_results=6
            )
            context = "\n\n".join([r['content'] for r in results['results'][:5]])
            reading_list = [{"title": r['title'], "url": r['url']} for r in results['results']]
        except Exception as e:
            context = f"Search Error: {e}"
            reading_list = []
        return {"raw_content": context, "reading_list": reading_list}

    def analyst_agent(state: AgentState):
        # CoT prompt
        prompt = f"""
        You are a Senior Strategy Consultant at McKinsey. 
        Your goal is to write a high-quality, data-driven industry report on: '{state['topic']}'.
        
        INSTRUCTIONS:
        1. Analyze the raw data to identify top critical trends.
        2. Structure your report using the MECE principle.
        3. Write in professional English Markdown.

        STRUCTURE:
        1. **Executive Summary** (Synthesize core insights)
        2. **Key Market Trends** (Data-driven bullet points)
        3. **Strategic Opportunities & Risks**

        Raw Data: {state['raw_content']}
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"structured_report": response.content}

    def tutor_agent(state: AgentState):
        prompt = f"""
        Create 3 multiple-choice questions based on the report below (IN ENGLISH).
        Format cleanly.
        Report: {state['structured_report']}
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"quiz": response.content}

    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("tutor", tutor_agent)
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "tutor")
    workflow.add_edge("tutor", END)
    
    return workflow.compile(), llm

# --- 8. main UI ---

st.title("🚀 SectorIQ")
st.subheader("The Enterprise Knowledge Accelerator")
st.markdown("Generate industry briefings, reading lists, and assessments in seconds.")

# Input (Container)
with st.container(border=True):
    col1, col2 = st.columns([3, 2])
    with col1:
        topic = st.text_input("Target Industry / Topic", placeholder="e.g. AI Agents in Finance")
    with col2:
        target_email = st.text_input("Recipient Email (Optional)", placeholder="name@company.com")
    
    start_btn = st.button("🚀 Start Workflow", type="primary", use_container_width=True)

# --- 9. logic ---

if start_btn:
    if not google_key or not tavily_key:
        st.error("⚠️ System Halted: Missing AI Keys. Please configure in Sidebar.")
    elif not topic:
        st.warning("⚠️ Please enter a topic.")
    else:
        # status bar
        status = st.status("🕵️ SectorIQ Agents Activated...", expanded=True)
        
        try:
            # 1. Initialize
            status.write(f"⚙️ Initializing **{model_choice}** swarm...")
            app, llm_instance = build_agent_graph(google_key, tavily_key, model_choice)
            
            # 2. Run Workflow
            status.write(f"🌍 Researcher Agent deep-diving into: **{topic}**...")
            result = app.invoke({"topic": topic})
            
            status.write("🧠 Analyst Agent synthesizing McKinsey-style report...")
            status.write("🎓 Tutor Agent preparing assessment...")
            
            # 3. QA Check
            status.write("⚖️ Chief Editor running Quality Assurance...")
            qa_result = run_qa_check(llm_instance, result['structured_report'], result['quiz'])
            
            status.update(label="✅ Workflow Complete!", state="complete", expanded=False)
            
            # --- result show ---
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Report", "📚 Reading List", "⚖️ QA Scorecard", "📤 Export & Email"])
            
            # Tab 1: report
            with tab1:
                st.markdown(result['structured_report'])
                st.divider()
                st.markdown("### 📝 Knowledge Check")
                st.markdown(result['quiz'])
            
            # Tab 2: read list
            with tab2:
                st.info("Curated sources for further reading:")
                if result['reading_list']:
                    for item in result['reading_list']:
                        st.markdown(f"🔗 **[{item['title']}]({item['url']})**")
                else:
                    st.write("No specific reading list found.")

            # Tab 3: dashboard
            with tab3:
                if qa_result:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Overall Score", f"{qa_result.get('overall_score', 0)}/10")
                    c2.metric("Depth", f"{qa_result.get('strategic_depth', 0)}/10")
                    c3.metric("Clarity", f"{qa_result.get('clarity', 0)}/10")
                    c4.metric("Relevance", f"{qa_result.get('quiz_relevance', 0)}/10")
                    st.success(f"💡 Editor's Feedback: {qa_result.get('recommendation', 'No feedback')}")
                else:
                    st.warning("QA Check unavailable.")

            # Tab 4: export
            with tab4:
                st.markdown("### Download Package")
                # PDF
                pdf_bytes = create_pdf(topic, result['structured_report'], result['quiz'], result['reading_list'])
                
                # button
                st.download_button(
                    label="📄 Download Full PDF",
                    data=pdf_bytes,
                    file_name=f"SectorIQ_{topic.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
                
                # send email
                st.divider()
                st.markdown("### Email Delivery")
                if target_email:
                    if email_user and email_pass:
                        if st.button("📧 Send PDF via Email"):
                            with st.spinner("Sending email..."):
                                success, msg = send_email_with_pdf(email_user, email_pass, target_email, topic, pdf_bytes)
                                if success:
                                    st.success(f"✅ {msg}")
                                else:
                                    st.error(f"❌ {msg}")
                    else:
                        st.warning("⚠️ To use Email, please configure 'Email Settings' in the Sidebar.")
                else:
                    st.info("Enter an email address above to enable sending.")

            # --- history ---
            summary_preview = result['structured_report'][:150] + "..."
            st.session_state.history.append({
                "topic": topic,
                "summary": summary_preview,
                "full_report": result['structured_report'],
                "quiz": result['quiz'],
                "reading_list": result['reading_list'],
                "qa_score": qa_result.get('overall_score', 'N/A') if qa_result else 'N/A',
                "timestamp": time.strftime("%H:%M")
            })
            
        except Exception as e:
            status.update(label="❌ Failed", state="error")
            st.error(f"Error: {str(e)}")
