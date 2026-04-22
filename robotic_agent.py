import os
import hashlib
import random
from typing import TypedDict

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Robotics Experiment Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Custom styling
# -----------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .hero {
            background: linear-gradient(135deg, rgba(78,115,223,0.12), rgba(90,200,250,0.10));
            border: 1px solid rgba(120,120,120,0.18);
            padding: 1.35rem 1.4rem;
            border-radius: 20px;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
            line-height: 1.1;
        }

        .hero-subtitle {
            font-size: 0.98rem;
            opacity: 0.78;
            margin-bottom: 0.25rem;
        }

        .pill {
            display: inline-block;
            padding: 0.3rem 0.65rem;
            margin-right: 0.45rem;
            margin-top: 0.35rem;
            border-radius: 999px;
            background: rgba(120,120,120,0.12);
            border: 1px solid rgba(120,120,120,0.14);
            font-size: 0.82rem;
        }

        .section-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(120,120,120,0.16);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            margin-bottom: 0.9rem;
        }

        .section-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .muted {
            opacity: 0.72;
            font-size: 0.92rem;
        }

        .small-label {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            opacity: 0.75;
            margin-bottom: 0.35rem;
        }

        .metric-box {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(120,120,120,0.15);
            border-radius: 18px;
            padding: 1rem;
            height: 100%;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.1;
            margin-top: 0.25rem;
        }

        .metric-caption {
            opacity: 0.72;
            font-size: 0.86rem;
            margin-top: 0.25rem;
        }

        .divider-space {
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
        }

        .stTextInput > div > div > input {
            border-radius: 14px;
            padding-top: 0.85rem;
            padding-bottom: 0.85rem;
        }

        div[data-testid="stButton"] button {
            border-radius: 14px;
            padding: 0.72rem 1.2rem;
            font-weight: 700;
        }

        .result-box {
            border-left: 4px solid rgba(90, 200, 250, 0.9);
            background: rgba(90, 200, 250, 0.06);
            padding: 0.9rem 1rem;
            border-radius: 14px;
            margin-top: 0.25rem;
        }

        .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.6rem;
            height: 1.6rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 800;
            margin-right: 0.45rem;
            background: rgba(120,120,120,0.16);
        }

        .footer-note {
            margin-top: 0.7rem;
            opacity: 0.68;
            font-size: 0.84rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# App header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🤖 Robotics Experiment Agent</div>
        <div class="hero-subtitle">
            A small agentic loop that plans an experiment, simulates a result, evaluates failure modes,
            and proposes the next iteration.
        </div>
        <div>
            <span class="pill">LangGraph workflow</span>
            <span class="pill">Gemini on free tier</span>
            <span class="pill">Research-style iteration</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Secrets / model
# -----------------------------
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Add GOOGLE_API_KEY to Streamlit secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
MODEL = "gemini-2.5-flash-lite"


# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict, total=False):
    goal: str
    plan: str
    result: str
    evaluation: str
    next_plan: str


# -----------------------------
# LLM helpers
# -----------------------------
@st.cache_resource

def get_llm():
    return ChatGoogleGenerativeAI(model=MODEL, temperature=0.2)


def llm_text(prompt: str) -> str:
    msg = HumanMessage(content=prompt)
    return get_llm().invoke([msg]).content


def plan_node(state: AgentState) -> dict:
    prompt = f"""
You are helping a robotics researcher.
Goal: {state['goal']}

Create a short experiment plan with:
- hypothesis
- parameters to test
- success metric
- likely failure modes

Keep it concise and practical.
"""
    return {"plan": llm_text(prompt)}


def simulate_node(state: AgentState) -> dict:
    # Deterministic-ish fake simulator so the demo stays cheap and repeatable.
    seed = int(hashlib.sha1(state["goal"].encode()).hexdigest(), 16)
    rng = random.Random(seed)

    score = rng.uniform(0.35, 0.92)
    failures = [
        "object slip during grasp",
        "camera noise caused bad perception",
        "policy overfit to easy cases",
        "trajectory was too aggressive",
        "reward was too sparse",
    ]
    failure = rng.choice(failures)
    confidence = rng.uniform(0.55, 0.93)

    result = f"""
Simulated success score: {score:.2f}
Confidence: {confidence:.2f}
What worked: the agent followed the proposed parameters.
What failed: {failure}.
Observation: performance dropped under harder conditions.
"""
    return {"result": result.strip()}


def evaluate_node(state: AgentState) -> dict:
    prompt = f"""
You are evaluating a robotics experiment.

Goal:
{state['goal']}

Plan:
{state['plan']}

Simulated result:
{state['result']}

Return:
1. main failure mode
2. why it happened
3. one concrete improvement for the next iteration

Use short, readable bullets.
"""
    return {"evaluation": llm_text(prompt)}


def refine_node(state: AgentState) -> dict:
    prompt = f"""
Improve the next robotics experiment.

Goal:
{state['goal']}

Previous plan:
{state['plan']}

Evaluation:
{state['evaluation']}

Write a better next plan.
Make exactly one major change from the previous plan.
Use concise bullet points.
"""
    return {"next_plan": llm_text(prompt)}


@st.cache_resource

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("simulate", simulate_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("refine", refine_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "simulate")
    builder.add_edge("simulate", "evaluate")
    builder.add_edge("evaluate", "refine")
    builder.add_edge("refine", END)
    return builder.compile()


graph = build_graph()


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.markdown("### Controls")
    st.caption("Use a robotics-style goal that has parameters and possible failure modes.")

    suggested_goals = [
        "Improve grasp success rate for a claw robot picking up objects of different shapes",
        "Reduce object slippage during robotic grasping",
        "Optimize pick-and-place accuracy under noisy camera input",
        "Improve robot arm trajectory for speed without losing precision",
    ]

    selected_goal = st.selectbox("Try a sample goal", ["Custom input"] + suggested_goals, index=0)

    st.markdown("---")
    st.markdown("### What this demo shows")
    st.write("- Planning")
    st.write("- Simulated experiment")
    st.write("- Failure analysis")
    st.write("- Next-iteration improvement")

    st.markdown("---")
    st.markdown("### Model")
    st.code(MODEL, language="text")


# -----------------------------
# Main input area
# -----------------------------
def apply_selected_goal():
    if selected_goal != "Custom input":
        st.session_state["goal_input"] = selected_goal


if "goal_input" not in st.session_state:
    st.session_state["goal_input"] = ""

if selected_goal != "Custom input" and st.session_state["goal_input"] == "":
    st.session_state["goal_input"] = selected_goal

col1, col2 = st.columns([3, 1])
with col1:
    goal = st.text_input(
        "Robotics goal",
        key="goal_input",
        placeholder="Improve pick-and-place success for a claw robot",
    )
with col2:
    st.write("")
    st.write("")
    run = st.button("Run agent", use_container_width=True)


# -----------------------------
# Results
# -----------------------------
if run and goal:
    with st.spinner("Running experiment loop..."):
        out = graph.invoke({"goal": goal})

    # Summary metrics
    score_text = "N/A"
    if "Simulated success score:" in out["result"]:
        try:
            score_text = out["result"].split("Simulated success score:")[1].split("\n")[0].strip()
        except Exception:
            score_text = "N/A"

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="small-label">Experiment status</div>
                <div class="metric-value">Completed</div>
                <div class="metric-caption">Plan → simulate → evaluate → refine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="small-label">Simulated score</div>
                <div class="metric-value">{score_text}</div>
                <div class="metric-caption">Deterministic-ish output for the same goal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="metric-box">
                <div class="small-label">Iteration depth</div>
                <div class="metric-value">1 loop</div>
                <div class="metric-caption">Enough to show agentic improvement</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Plan", "Result", "Evaluation", "Next Iteration"])

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="step-number">1</span>Experiment plan</div>', unsafe_allow_html=True)
        st.write(out["plan"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="step-number">2</span>Simulated result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-box"><pre style="white-space: pre-wrap; margin: 0;">{out["result"]}</pre></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="step-number">3</span>Evaluation</div>', unsafe_allow_html=True)
        st.write(out["evaluation"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="step-number">4</span>Next iteration</div>', unsafe_allow_html=True)
        st.write(out["next_plan"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### What to say in your demo")
    st.info(
        "This agent simulates a robotics research loop: it proposes an experiment, checks the outcome, identifies the failure mode, and suggests one better next step."
    )

    with st.expander("Show raw output"):
        st.json(out)

    st.caption("Tip: use a goal like 'Reduce object slippage during robotic grasping' for a cleaner demo.")

else:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">How it works</div>
            <div class="muted">
                Enter a robotics goal, run the agent, and watch it move through four stages:
                planning, simulated execution, evaluation, and refinement.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        st.markdown(
            """
            <div class="metric-box">
                <div class="small-label">Best use case</div>
                <div class="metric-value">Research loop</div>
                <div class="metric-caption">Looks closest to agentic robotics experimentation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with demo_col2:
        st.markdown(
            """
            <div class="metric-box">
                <div class="small-label">Design choice</div>
                <div class="metric-value">No embeddings</div>
                <div class="metric-caption">This demo focuses on iteration, not retrieval</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='footer-note'>Try one of the sample goals in the sidebar, then run the agent.</div>",
        unsafe_allow_html=True,
    )
