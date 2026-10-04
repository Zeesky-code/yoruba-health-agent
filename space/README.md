---
title: Yorùbá Health Agent
emoji: 🩺
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Yorùbá health Q&A from MedlinePlus, with citations
---

# Yorùbá Health Information Agent

Ask a health question in Yorùbá. A Cohere tool-use agent searches English MedlinePlus
pages, answers in Yorùbá with sources, and refuses off-topic questions or requests for
personal medical advice. Every agent step is shown in the trace panel.

**Not medical advice.** This is a research prototype that has not been clinically
validated.

Code, eval set, results and write-up: https://github.com/Zeesky-code/yoruba-health-agent
