import streamlit as st
import os
import base64
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

# --- 1. page ---
st.set_page_config(
    page_title="SectorIQ - Enterprise Agent",
    page_icon="🚀",
    layout="wide"
)

# --- 2. sidebar and API ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083256.png", width=50)
    st.title("SectorIQ")
    st.markdown("### ⚙️ Configuration")

    # user can input their own KEYs
    google_key = st.text_input("Google API Key", type="password", help="Get from aistudio.google.com")
    tavily_key = st.text_input("Tavily API Key", type="password", help="Get from tavily.com")

    st.markdown("---")
    st.info("💡 **Enterprise Mode:** Generates McKinsey-style reports & employee training quizzes.")


# --- 3. core functions ---

# status
class AgentState(TypedDict):
    topic: str
    raw_content: str
    structured_report: str
    quiz: str


def build_agent_graph(api_key_google, api_key_tavily):
    """build agent based on users' api key"""

    # innitial model, test with gemini-2.5-flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key_google
    )
    tavily = TavilyClient(api_key=api_key_tavily)

    # node function
    def researcher_agent(state: AgentState):
        try:
            results = tavily.search(query=f"{state['topic']} key trends market size 2025", search_depth="basic")
            context = "\n\n".join([r['content'] for r in results['results'][:3]])
        except Exception as e:
            context = f"Search Error: {e}"
        return {"raw_content": context}

    def analyst_agent(state: AgentState):
        prompt = f"""
        You are a Senior Consultant at McKinsey. Analyze: '{state['topic']}'.
        Output a structured Markdown report:
        1. Executive Summary
        2. Key Trends
        3. Strategic Risks

        Data: {state['raw_content']}
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"structured_report": response.content}

    def tutor_agent(state: AgentState):
        prompt = f"""
        Create 3 multiple-choice questions based on the report below.
        Format cleanly.
        Report: {state['structured_report']}
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"quiz": response.content}

    # build gragh
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("tutor", tutor_agent)

    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "tutor")
    workflow.add_edge("tutor", END)

    return workflow.compile()


# --- 4. main view UI ---

st.title("🚀 SectorIQ: The Enterprise Knowledge Accelerator")
st.markdown("#### Automate industry research & employee onboarding in seconds.")

# INPUT here
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input("Enter an Industry or Topic:", placeholder="e.g. Autonomous Agents in Healthcare")
with col2:
    st.write("")  # Spacer
    st.write("")  # Spacer
    start_btn = st.button("🚀 Start Workflow", use_container_width=True, type="primary")

# --- 5. logic ---
if start_btn:
    if not google_key or not tavily_key:
        st.error("⚠️ Please enter your API Keys in the sidebar first!")
    elif not topic:
        st.warning("⚠️ Please enter a topic.")
    else:
        # status bar
        status = st.status("🕵️ SectorIQ Agents are working...", expanded=True)

        try:
            # 1. initial
            status.write("⚙️ Initializing Agent Team...")
            app = build_agent_graph(google_key, tavily_key)

            # 2. research
            status.write(f"🌍 Researcher Agent is scanning the web for '{topic}'...")
            result = app.invoke({"topic": topic})

            # 3. analysis
            status.write("🧠 Analyst Agent is synthesizing the report...")

            # 4. finish
            status.update(label="✅ Workflow Complete!", state="complete", expanded=False)

            # --- result show ---
            tab1, tab2, tab3 = st.tabs(["📊 Executive Report", "📝 Onboarding Quiz", "📥 Export"])

            with tab1:
                st.markdown(result['structured_report'])

            with tab2:
                st.info("Test your knowledge based on the report above.")
                st.markdown(result['quiz'])
                if st.button("Submit Answers"):
                    st.balloons()
                    st.success("Results recorded!")

            with tab3:
                # preparing content for download
                full_text = f"# SectorIQ Report: {topic}\n\n{result['structured_report']}\n\n---\n\n## Quiz\n{result['quiz']}"
                st.download_button(
                    label="📥 Download Full Report (.md)",
                    data=full_text,
                    file_name=f"SectorIQ_{topic.replace(' ', '_')}.md",
                    mime="text/markdown"
                )

        except Exception as e:
            status.update(label="❌ Error occurred", state="error")

            st.error(f"An error occurred: {str(e)}")
