<table>
  <tr>
    <td width="140" align="center" valign="middle">
      <img src="https://macos-typora-imagebed.oss-cn-beijing.aliyuncs.com/imgmacos/202604231412945.png" alt="AIED 2026 Seoul emblem" width="110">
    </td>
    <td align="left" valign="middle">
      <strong>AIED 2026</strong><br>
      <sub>Seoul, Republic of Korea</sub>
    </td>
  </tr>
</table>

<h1 align="center">Tell Me Why: Designing an Explainable LLM-based Dialogue System for Student Problem Behavior Diagnosis</h1>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><strong>Paper</strong></a> ·
  <a href="#citation"><strong>Citation</strong></a>
</p>

<p align="center">
  Accepted at <strong>AIED 2026</strong><br>
  <em>27th International Conference on Artificial Intelligence in Education</em><br>
  Seoul, Republic of Korea · June 29–July 3, 2026
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Conference-AIED%202026-orange">
  <img src="https://img.shields.io/badge/Status-Accepted-success">
  <img src="https://img.shields.io/badge/License-MIT-blue">
</p>


---

## Overview

This repository accompanies our paper **“Tell Me Why: Designing an Explainable LLM-based Dialogue System for Student Problem Behavior Diagnosis”**, accepted at **AIED 2026**.

Diagnosing student problem behaviors requires teachers to synthesize multifaceted information, identify behavioral categories, and plan intervention strategies. We present an **explainable dialogue system** built on a fine-tuned large language model (LLM) for student problem behavior diagnosis. Beyond generating recommendations through multi-turn dialogue, our system uses a **hierarchical attribution method** to identify dialogue evidence supporting each recommendation and produce natural-language explanations grounded in that evidence.

Our results show that the proposed attribution method outperforms baseline approaches in identifying supporting evidence. In a preliminary user study with 22 pre-service teachers, participants who received explanations reported higher trust in the system.

<details>
<summary><strong>Abstract</strong></summary>

Diagnosing student problem behaviors requires teachers to synthesize multifaceted information, identify behavioral categories, and plan intervention strategies. Although fine-tuned large language models (LLMs) can support this process through multi-turn dialogue, they rarely explain why a strategy is recommended, limiting transparency and teachers' trust. To address this issue, we present an explainable dialogue system built on a fine-tuned LLM. The system uses a hierarchical attribution method based on explainable AI (xAI) to identify dialogue evidence for each recommendation and generate a natural-language explanation based on that evidence. In technical evaluation, the method outperformed baseline approaches in identifying supporting evidence. In a preliminary user study with 22 pre-service teachers, participants who received explanations reported higher trust in the system. These findings suggest a promising direction for improving LLM explainability in educational dialogue systems.

</details>

---

## Repository Structure

```text
TwoLevelAttribution/
├── two_level_attribution/
│   ├── __init__.py
│   ├── dialogue_attributor.py    # Core hierarchical attribution
│   ├── tree.py                   # Tree data structure
│   ├── context_ops.py            # Context operations
│   ├── kvcache.py                # KV cache for efficiency
│   └── model_utils.py            # Model utilities
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                # Hit@k, MRR, etc.
├── examples/
│   └── example_usage.py          # Usage examples
├── requirements.txt
└── README.md
