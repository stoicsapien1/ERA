# 📖 My Journey Through ERA V4: The Dawn of Transformers

This repository documents my learning journey through the **ERA - The Dawn of Transformers, Version 4** course.  
It serves as a personal archive of:
- ✍️ Handwritten notes (scanned PDFs)
- 💻 Jupyter notebooks (implementations & experiments)
- 📚 Supplementary references (papers, articles, and links)

---

## 🚀 About the Course: ERA V4

**ERA V4** is an ambitious curriculum designed for those ready to train, optimize, and understand AI systems at scale.  
The course emphasizes **real-world engineering challenges**, including:

- **Full-Scale LLM Training** → Pre-training a **70B parameter LLM** from scratch, followed by instruction tuning.  
- **CoreSet Thinking** → Data efficiency techniques for compact dataset representation.  
- **Multi-GPU Training** → Training **ResNet-50 on full ImageNet** using distributed training.  
- **Quantization-Aware Training (QAT)** → Training models in quantized mode, not just fine-tuning.  
- **Modern Modalities** → Balanced coverage of **Vision, Language, RL, and Embeddings**, with deployment in mind.  

---

## 📂 Repository Structure

Each session has a dedicated folder containing:

- ✍️ **Notes** → Handwritten notes (PDF) summarizing key concepts.  
- 💻 **Code** → Jupyter notebooks with implementations & experiments.  
- 📚 **Extras** → External references, research papers, and additional readings.  

---

## 📅 Course Syllabus

The course spans **20 intensive sessions**, progressing from fundamentals to training large-scale models.  

<details>
<summary><strong>Session 1: Introduction to AI & Neural Networks</strong></summary>

- What is AI? Evolution and applications  
- Fundamentals: perceptrons, activations, weights, bias  
- Course overview: journey to training a **70B LLM**  
- Dev setup: Python, VS Code, CUDA drivers  
- Tools: PyTorch, WandB, Git, Cursor  

</details>

<details>
<summary><strong>Session 2: Python, Git, and Web Basics</strong></summary>

- Python essentials for ML  
- Git/GitHub workflow: branching, merging, collaboration  
- Web basics: HTML/CSS/JS + Flask  
- Objective: Launch a web UI for visualizing model outputs  

</details>

<details>
<summary><strong>Session 3: PyTorch Fundamentals & AWS EC2</strong></summary>

- Tensors & operations in PyTorch  
- AutoGrad & computational graphs  
- Building simple NNs with training/validation loops  
- Cloud setup: EC2 instance + SSH  

</details>

<details>
<summary><strong>Session 4: First Neural Network on Cloud</strong></summary>

- Build & train MLP on **MNIST**  
- Visualize loss curves using **WandB**  
- Training on Colab & AWS EC2  
- Save/load model checkpoints  
- Flask API for predictions  

</details>

<details>
<summary><strong>Session 5: CNNs & Backpropagation</strong></summary>

- CNN basics: convolution, filters, receptive fields  
- Implementing & training CNNs in PyTorch  
- Backpropagation explained  
- Techniques for effective training  

</details>

<details>
<summary><strong>Session 6–8: Advanced CNNs & CoreSets</strong></summary>

- Hands-on CNN coding practice (VGG, Inception)  
- Data augmentation: CutOut, MixUp, RICAP  
- Normalization & regularization techniques  
- ResNets & One Cycle Policy  
- CoreSets for dataset efficiency  

</details>

<details>
<summary><strong>Session 9: Multi-GPU Training on ImageNet</strong></summary>

- Distributed Data Parallel (DDP) in PyTorch  
- Training **ResNet-50 on full ImageNet**  
- AWS multi-GPU training setup  
- Performance visualization  

</details>

<details>
<summary><strong>Session 10–13: Transformers & LLMs</strong></summary>

- Transformers: self-attention, MHA, positional encoding  
- Implementing transformer blocks from scratch  
- Tokenization & embeddings (BPE, t-SNE/UMAP)  
- GPT-style architectures, RoPE embeddings  
- Mixed-precision training (FP16/BF16)  
- LLM evaluation: perplexity, BLEU, MMLU  

</details>

<details>
<summary><strong>Session 14: Quantization-Aware Training (QAT)</strong></summary>

- QAT for **pre-training**, not just fine-tuning  
- Implementing QAT in PyTorch  
- Monitoring with WeightWatcher  

</details>

<details>
<summary><strong>Session 15–18: Multimodal & RL</strong></summary>

- **CLIP**: Vision-Language models (image + text)  
- Reinforcement Learning: Q-learning, PPO, A3C  
- Continuous action spaces: DDPG, PPO  
- RLHF pipeline: reward modeling, PPO, alignment  

</details>

<details>
<summary><strong>Session 19: Training a 70B LLM</strong></summary>

- Full pre-training workflow (context length, gradient checkpointing)  
- Model parallelism strategies  
- Instruction tuning & deployment with **vLLM**  

</details>

<details>
<summary><strong>Session 20: Capstone Project</strong></summary>

- Full-stack AI project (model training + deployment)  
- Deliverables: demo, write-up, GitHub repo  

</details>

---

## 🛠️ Key Technologies & Concepts

**Frameworks & Libraries**: PyTorch, vLLM  
**Platforms**: AWS EC2, Google Colab  
**Tools**: Git, GitHub, WandB, VS Code, Cursor  

**Core Concepts**:
- Neural Networks (MLP, CNNs, ResNet)  
- Transformers (ViT, GPT)  
- Large Language Models (LLMs)  
- Reinforcement Learning (Q-Learning, PPO, RLHF)  
- Vision-Language Models (CLIP)  
- Quantization-Aware Training (QAT)  
- CoreSet Sampling  
- Distributed Training (DDP)  

---

## 🔄 ERA V3 → V4: What’s New?

ERA V4 represents a **shift from theory to practice at scale**:

- Training a **70B LLM** (vs small models in V3)  
- CoreSets as a primary method (not just data cleaning)  
- Training **ResNet-50 on full ImageNet** with multi-GPU setup  
- **QAT as a core skill** (not LoRA/PEFT shortcuts)  
- **vLLM for production inference**  
- RLHF & alignment included early  
- Cleaner scope: focus on CNNs, Transformers, RL, and LLMs at scale  

---

## ⚠️ Disclaimer

This repository is a **personal learning archive** for the ERA V4 course.  
It is **not an official course resource**.  
All materials are for **educational purposes only**.

---
