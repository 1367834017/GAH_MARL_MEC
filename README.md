# GAH-MARL-MEC

Official implementation and simulation environment for **GAH-MARL**.

This repository contains the official code and simulation environment for the paper:

> **Heterogeneous Information Fusion in Graph-Based Asynchronous Hierarchical Multi-Agent Reinforcement Learning for Hybrid-Action Task Offloading**

---

## Overview

GAH-MARL is a graph-based asynchronous hierarchical multi-agent reinforcement learning framework for hybrid-action task offloading in mobile edge computing (MEC).  
The proposed method combines graph-based communication and hierarchical decision-making to improve cooperation among user equipments (UEs) under dynamic network conditions.

This repository provides:

- the implementation of the proposed GAH-MARL framework,
- the simulation environment for MEC task offloading,
- training and evaluation scripts.

---

## Repository Structure

- `envs/`: MEC simulation environment
- `src/agents/`: agent implementation
- `src/utils/`: utility modules such as normalization and replay buffer
- `src/train.py`: training script
- `src/test.py`: testing script



