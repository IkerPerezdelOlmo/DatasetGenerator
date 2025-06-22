# Graph Dataset Generator

This repository contains tools to generate and preprocess graph data suitable for models such as **LGGM** and **GraphRNN**.

## Overview

The codebase includes the following main scripts:

### `grafo_artifizial_sortzailea.py`

Generates synthetic graphs from well-known distributions, including:

- **Erdős–Rényi (ER)**
- **Community**

These graphs are useful for benchmarking generative models.

---

### `convertFileIntoGraphs.py`

Processes large real-world graphs ( from the [DIMACS](https://dimacs.rutgers.edu/programs/challenge/) dataset) and **samples smaller subgraphs** from them. 

> ⚠️ **Note:** The original large graph files are not included in this repository due to GitHub's file size limitations.  
You can download them manually from official sources.

- [**Internet Topology** link](https://snap.stanford.edu/data/as-Skitter.html)
- [**New York Roads** link](http://www.dis.uniroma1.it/~challenge9/download.shtml)

---

### `LGGM/TEST/modify_data.py`
Processes the created graphs to adjust the format to LGGM model 

