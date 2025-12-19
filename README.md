# 🕸️ AI Graph Search Visualizer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualize-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **A comprehensive tool to visualize and compare Artificial Intelligence search algorithms (DFS, BFS, UCS, A*) on random graphs.**

---

## 📖 Overview
This project is an interactive web application designed to demonstrate how different AI search algorithms navigate through a graph. It allows users to generate random graphs (Directed/Undirected), visualize the search process step-by-step, and compare the performance of different algorithms in terms of **Cost**, **Steps**, and **Visited Nodes**.

Designed as a Master's project to showcase the practical implementation of search strategies in Artificial Intelligence.

## 🚀 Features

* **🎲 Dynamic Graph Generation:** Create random graphs with customizable nodes and connectivity.
* **🔄 Directed & Undirected Modes:** Support for both one-way and two-way edge connections.
* **🧠 Multiple Algorithms:** Implementation of 4 core search strategies.
* **📊 Comparison Mode:** Run all algorithms simultaneously to compare efficiency side-by-side.
* **🎨 Interactive Visualization:** Beautiful plotting using Matplotlib with support for custom layouts.
* **⚙️ Customizable Settings:** Toggle edge weights, change start/goal nodes, and more.

---

## 🤖 Algorithms Implemented

| Algorithm | Type | Description | Optimality |
| :--- | :---: | :--- | :---: |
| **DFS** (Depth-First Search) | Uninformed | Explores as deep as possible along each branch before backtracking. | ❌ No |
| **BFS** (Breadth-First Search) | Uninformed | Explores all neighbor nodes at the present depth prior to moving on to the nodes at the next depth level. | ✅ Yes (in unweighted) |
| **UCS** (Uniform Cost Search) | Uninformed | Expands the least-cost node. Guaranteed to find the cheapest path. | ✅ Yes |
| **A\*** (A-Star Search) | Informed | Uses a heuristic function (Euclidean distance) to estimate the cost to the goal. Efficient and optimal. | ✅ Yes |

---

## 🛠️ Installation & Usage

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hzabbas/graph-search-ai.git
    cd graph-search-ai
    ```

2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

4.  **Open your browser:**
    The app should automatically open at `http://localhost:8501`.

---

## ☁️ Deployment

This app is ready to be deployed on **Streamlit Community Cloud**:

1.  Push the code to GitHub.
2.  Go to [share.streamlit.io](https://share.streamlit.io).
3.  Connect your repository and deploy!

---

## 👨‍💻 Author

**Abbas Hajizadeh**
* B.Sc. in Computer Science

---

<div align="center">
    <i>Built with ❤️ using Python & Streamlit</i>
</div>
