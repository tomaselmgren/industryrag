import os
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64

plt.switch_backend('Agg') 

class Visualizer():
    def __init__(self):
        self

    def build_directory_graph(self, root_dir, graph=None, parent_node=None):
        if graph is None:
            graph = nx.DiGraph()

        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            graph.add_node(item_path)
            if parent_node:
                graph.add_edge(parent_node, item_path)

            if os.path.isdir(item_path):
                self.build_directory_graph(item_path, graph, item_path)

        return graph


    def generate_directory_structure_image_with_highlight(self, root_dir, highlight_path):
        graph = self.build_directory_graph(root_dir)
        pos = nx.spring_layout(graph, seed=42)  # Calculate positions for all nodes
        
        plt.figure(figsize=(12, 8))
        
        # Separate highlight path and normal nodes
        highlight_nodes = []
        highlight_edges = []
        current_path = root_dir
        
        # Build the highlight path based on the input
        for part in highlight_path.split('/'):
            current_path = os.path.join(current_path, part)
            highlight_nodes.append(current_path)
            if len(highlight_nodes) > 1:
                highlight_edges.append((highlight_nodes[-2], highlight_nodes[-1]))

        # Draw unhighlighted nodes (fainter and without labels)
        unhighlighted_nodes = [node for node in graph.nodes() if node not in highlight_nodes]
        nx.draw(graph, pos, nodelist=unhighlighted_nodes, with_labels=False, node_size=50, 
                node_color="lightgray", edge_color="lightgray", alpha=0.5)

        # Draw highlighted nodes and edges (more prominent)
        nx.draw_networkx_nodes(graph, pos, nodelist=highlight_nodes, node_size=300, node_color="red")
        nx.draw_networkx_edges(graph, pos, edgelist=highlight_edges, width=4, edge_color="red")
        
        # Adjust the label positions to avoid overlap with nodes and edges
        label_pos = {key: (value[0] + 0.03, value[1] + 0.03) for key, value in pos.items()}  # Offset by adjusting x and y coordinates

        # Add labels only to highlighted nodes
        labels = {node: os.path.basename(node) for node in highlight_nodes}
        nx.draw_networkx_labels(graph, label_pos, labels=labels, font_size=15, font_family="sans-serif")

        plt.title(f"Directory Structure of {os.path.basename(root_dir)} with Highlighted Path")
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return image_base64