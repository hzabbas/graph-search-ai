import streamlit as st
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Graph Search Project", layout="wide")
st.title("AI Project: Graph Search")
st.write("Abbas Hajizadeh")

class GraphProject:
    def __init__(self, num_nodes=30, directed=False):
        self.num_nodes = num_nodes
        self.directed = directed
        self.matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        self.adj_list = {i: [] for i in range(num_nodes)}
        self.positions = {} 
        self.edges = [] 
        
        self.generate_random_matrix()
        self.process_geometry()

    def generate_random_matrix(self):
        # Probability of edge creation
        connection_prob = 0.06
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    if random.random() < connection_prob:
                        self.matrix[i][j] = 1
                        if not self.directed:
                            self.matrix[j][i] = 1
        
        # Ensure the graph is connected (Ring structure)
        nodes = list(range(self.num_nodes))
        random.shuffle(nodes)
        for k in range(len(nodes)):
            u = nodes[k]
            v = nodes[(k + 1) % len(nodes)]
            self.matrix[u][v] = 1
            if not self.directed:
                self.matrix[v][u] = 1

    def process_geometry(self):
        # Grid layout to prevent node overlap
        cols = 6
        rows = 5
        cell_width = 100 / cols
        cell_height = 100 / rows
        
        grid_cells = []
        for r in range(rows):
            for c in range(cols):
                grid_cells.append((r, c))
        random.shuffle(grid_cells)

        for i in range(self.num_nodes):
            row, col = grid_cells[i]
            base_x = col * cell_width
            base_y = row * cell_height
            # Add random jitter within the cell
            x = base_x + random.uniform(5, cell_width - 5)
            y = base_y + random.uniform(5, cell_height - 5)
            self.positions[i] = (x, y)

        self.edges = [] 
        self.adj_list = {i: [] for i in range(self.num_nodes)}

        # Calculate weights based on Euclidean distance
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.matrix[i][j] == 1:
                    x1, y1 = self.positions[i]
                    x2, y2 = self.positions[j]
                    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    weight = round(dist, 1)
                    
                    self.adj_list[i].append((j, weight))
                    self.edges.append((i, j, weight))

    # ----------------
    # Search Algorithms
    # ----------------

    def dfs(self, start, goal):
        # Stack implementation for DFS
        stack = [(start, [start], 0)]
        visited = set()
        visited_count = 0
        while stack:
            current, path, cost = stack.pop()
            if current in visited: continue
            visited.add(current)
            visited_count += 1
            if current == goal: return path, cost, visited_count
            
            for neighbor, weight in self.adj_list[current]:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], cost + weight))
        return None, 0, visited_count

    def bfs(self, start, goal):
        # Queue implementation for BFS
        queue = [(start, [start], 0)]
        visited = set()
        visited_count = 0
        while queue:
            current, path, cost = queue.pop(0)
            if current in visited: continue
            visited.add(current)
            visited_count += 1
            if current == goal: return path, cost, visited_count
            
            for neighbor, weight in self.adj_list[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor], cost + weight))
        return None, 0, visited_count

    def ucs(self, start, goal):
        pq = [(0, start, [start])]
        visited = set()
        visited_count = 0
        while pq:
            # Sort by cost (lowest first)
            pq.sort(key=lambda x: x[0])
            cost, current, path = pq.pop(0)
            if current in visited: continue
            visited.add(current)
            visited_count += 1
            if current == goal: return path, cost, visited_count
            
            for neighbor, weight in self.adj_list[current]:
                if neighbor not in visited:
                    pq.append((cost + weight, neighbor, path + [neighbor]))
        return None, 0, visited_count

    def heuristic(self, u, v):
        # Euclidean distance heuristic
        x1, y1 = self.positions[u]
        x2, y2 = self.positions[v]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def a_star(self, start, goal):
        start_h = self.heuristic(start, goal)
        pq = [(start_h, 0, start, [start])] # (f, g, node, path)
        visited = set()
        visited_count = 0
        while pq:
            # Sort by f = g + h
            pq.sort(key=lambda x: x[0])
            f, g, current, path = pq.pop(0)
            if current in visited: continue
            visited.add(current)
            visited_count += 1
            if current == goal: return path, g, visited_count
            
            for neighbor, weight in self.adj_list[current]:
                if neighbor not in visited:
                    new_g = g + weight
                    new_h = self.heuristic(neighbor, goal)
                    pq.append((new_g + new_h, new_g, neighbor, path + [neighbor]))
        return None, 0, visited_count

def plot_graph(graph_obj, path=None, show_weights=True):
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Transparent background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    for u, v, w in graph_obj.edges:
        x1, y1 = graph_obj.positions[u]
        x2, y2 = graph_obj.positions[v]
        
        color = 'gray'
        width = 0.5
        alpha = 0.3
        is_path_edge = False
        
        # Highlight path
        if path:
            for k in range(len(path) - 1):
                if graph_obj.directed:
                    if path[k] == u and path[k+1] == v:
                        color = 'red'; width = 2.5; alpha = 1
                        is_path_edge = True
                        break
                else:
                    if (path[k] == u and path[k+1] == v) or (path[k] == v and path[k+1] == u):
                        color = 'red'; width = 2.5; alpha = 1
                        is_path_edge = True
                        break
        
        # Draw edges (Arrow or Line)
        if graph_obj.directed:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color=color, 
                                        lw=width, alpha=alpha, mutation_scale=15,
                                        shrinkA=15, shrinkB=15),
                        zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], c=color, linewidth=width, alpha=alpha, zorder=1)
        
        # Display weights
        if show_weights or is_path_edge:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            txt_color = 'darkred' if is_path_edge else 'darkblue'
            txt_weight = 'bold' if is_path_edge else 'normal'
            
            ax.text(mid_x, mid_y, str(w), fontsize=8, color=txt_color, weight=txt_weight,
                    ha='center', va='center', 
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.2))

    # Draw nodes
    for node, (x, y) in graph_obj.positions.items():
        c = '#2E8B57' 
        if path:
            if node in path: c = '#90EE90'
            if node == path[0]: c = 'yellow'
            if node == path[-1]: c = 'orange'
            
        ax.scatter(x, y, s=500, c=c, edgecolors='black', linewidth=1, zorder=2)
        ax.text(x, y, str(node), zorder=3, ha='center', va='center', fontsize=10, weight='bold')
    
    ax.axis('off')
    return fig

# ----------------
# Sidebar
# ----------------
st.sidebar.header("⚙️ Graph Settings")

is_directed = st.sidebar.checkbox("Directed Graph", value=False)
show_weights = st.sidebar.checkbox("Show Edge Weights", value=True)

if st.sidebar.button("Generate New Graph 🎲"):
    st.session_state['graph'] = GraphProject(directed=is_directed)
    st.session_state['result'] = None
    st.session_state['comparison'] = None
    st.success(f"New graph generated.")

if 'graph' not in st.session_state:
    st.session_state['graph'] = GraphProject(directed=is_directed)
graph = st.session_state['graph']

if graph.directed != is_directed:
    st.warning("Graph type changed. Click 'Generate New Graph'.")

st.sidebar.markdown("---")
st.sidebar.header("🔍 Search Settings")
algo = st.sidebar.selectbox("Algorithm", ["DFS", "BFS", "UCS", "A*"])
start_node = st.sidebar.number_input("Start Node", 0, 29, 0)
goal_node = st.sidebar.number_input("Goal Node", 0, 29, 15)

col_s1, col_s2 = st.sidebar.columns(2)

with col_s1:
    if st.button("🚀 Start Search"):
        path, cost, visited = None, 0, 0
        
        if algo == "DFS": path, cost, visited = graph.dfs(start_node, goal_node)
        elif algo == "BFS": path, cost, visited = graph.bfs(start_node, goal_node)
        elif algo == "UCS": path, cost, visited = graph.ucs(start_node, goal_node)
        elif algo == "A*": path, cost, visited = graph.a_star(start_node, goal_node)
            
        st.session_state['result'] = (path, cost, visited, algo)
        st.session_state['comparison'] = None

with col_s2:
    if st.button("📊 Compare All"):
        results = []
        algs = ["DFS", "BFS", "UCS", "A*"]
        
        for name in algs:
            p, c, v = None, 0, 0
            if name == "DFS": p, c, v = graph.dfs(start_node, goal_node)
            elif name == "BFS": p, c, v = graph.bfs(start_node, goal_node)
            elif name == "UCS": p, c, v = graph.ucs(start_node, goal_node)
            elif name == "A*": p, c, v = graph.a_star(start_node, goal_node)
            
            steps = len(p) - 1 if p else 0
            found = "✅ Yes" if p else "❌ No"
            results.append({"Algorithm": name, "Path Cost": round(c, 1), "Steps": steps, "Visited Nodes": v, "Found": found})
        
        st.session_state['comparison'] = pd.DataFrame(results)
        st.session_state['result'] = None

# --- 
# Main Display 
# ---
st.markdown("---")

if 'comparison' in st.session_state and st.session_state['comparison'] is not None:
    st.subheader("🏆 Performance Comparison")
    st.table(st.session_state['comparison'])
    st.subheader("Graph Structure")
    st.pyplot(plot_graph(graph, None, show_weights))

elif 'result' in st.session_state and st.session_state['result']:
    path, cost, visited, name = st.session_state['result']
    
    if path:
        st.success(f"✅ Path found using {name}!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Path Cost", f"{cost:.1f}")
        c2.metric("Nodes Visited", visited)
        c3.metric("Steps", len(path)-1)
        st.write(f"**Path:** {path}")
        st.pyplot(plot_graph(graph, path, show_weights))
    else:
        st.error("❌ No path found.")
        st.pyplot(plot_graph(graph, None, show_weights))

else:
    st.info("Choose an action from the sidebar.")
    st.pyplot(plot_graph(graph, None, show_weights))