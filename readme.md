# Robotics Experiment Agent

A lightweight agentic AI system that simulates the robotics research loop:

> Plan → Execute → Evaluate → Improve

Built using LangGraph, Gemini (free tier), and Streamlit, this project demonstrates how autonomous agents can iteratively design and refine experiments without human intervention.

---

## Demo Overview

This app allows a user to input a robotics goal (e.g., improving grasp success rate), and the agent will:

1. Generate an experiment plan
2. Simulate experiment results
3. Identify failure modes
4. Propose an improved next iteration

This mimics how real robotics researchers run experiments and refine systems.

---

## Why this project

Most AI demos are:
- Chatbots
- RAG systems

This project focuses on:

- Agentic workflows
- Failure-aware reasoning
- Iterative improvement loops

These are core to building autonomous systems.

---

## Architecture

The system is built as a LangGraph workflow:

```
User Goal
   ↓
[Planner]
   ↓
[Simulator]
   ↓
[Evaluator]
   ↓
[Refiner]
   ↓
Next Iteration
```

### Components

- Planner: Generates experiment hypothesis and parameters
- Simulator: Produces deterministic simulated results
- Evaluator: Identifies failure modes
- Refiner: Improves experiment design

---

## Tech Stack

- Frontend: Streamlit
- LLM: Gemini (free tier)
- Framework: LangChain + LangGraph
- Language: Python

---

## Installation

### 1. Clone the repo

```
git clone <your-repo-url>
cd robotics-agent
```

### 2. Create virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Setup API Key

Create the following file:

```
.streamlit/secrets.toml
```

Add:

```
GOOGLE_API_KEY = "your_api_key_here"
```

You can get a free Gemini API key from Google AI Studio.

---

## Run the App

```
streamlit run app.py
```

If that doesn’t work:

```
python -m streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## Example Inputs

Try goals like:

- Improve grasp success rate for a claw robot
- Reduce object slippage during robotic grasping
- Optimize pick-and-place accuracy under noisy vision
- Improve robot arm trajectory for speed and precision

---

## What Makes This "Agentic"

This project demonstrates key agent behaviors:

- Multi-step reasoning
- Modular workflow design
- Self-evaluation
- Iterative refinement

Unlike static LLM calls, the system loops and improves its output.

---

## Why No Embeddings?

This system focuses on decision-making and iteration, not retrieval.

Embeddings would be useful for:
- Long-term experiment memory
- Knowledge base retrieval

They are intentionally excluded to keep the focus on agent loops.

---

## Future Improvements

- Add experiment memory (vector database)
- Integrate with real robotics simulators
- Support multi-step iterative loops
- Add tool integrations (logging, visualization, parameter tuning)

---

## Key Insight

The most important part of autonomous AI systems is not generation, but evaluation and iteration.

---

## One-Line Summary (for resume)

Built an agentic AI system that simulates robotics experimentation by planning experiments, identifying failure modes, and iteratively improving results using LangGraph and Gemini.

---

## Acknowledgements

- LangGraph for agent workflows
- Google Gemini for LLM access
- Streamlit for rapid UI prototyping

---

## Contact

If you found this useful or want to collaborate, feel free to reach out.

