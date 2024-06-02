import networkx as nx
import matplotlib.pyplot as plt
import spacy


class TextGraphBuilder:
    def __init__(self):
        # Load the English language model in spaCy
        self.nlp = spacy.load("en_core_web_sm")

    def build_graph_with_attributes(self, text):
        # Parse the text using spaCy
        doc = self.nlp(text)

        # Initialize a directed graph
        graph = nx.DiGraph()

        # Initialize a dictionary to store word frequencies
        word_freq = {}

        # Iterate over the tokens in the parsed document
        for i, token in enumerate(doc):
            # Increment word frequency count
            word_freq[token.text] = word_freq.get(token.text, 0) + 1

            # Add the token as a node to the graph
            if not graph.has_node(token.text):
                graph.add_node(token.text)

            # Add edges between consecutive tokens
            if i > 0:
                graph.add_edge(doc[i - 1].text, token.text)

        # Add word frequency as node attribute
        nx.set_node_attributes(graph, word_freq, 'frequency')

        # If word embeddings are available, add them as node attributes
        for token in doc:
            if token.has_vector:
                embeddings = token.vector.tolist()  # Convert embeddings to list
                graph.nodes[token.text]['embeddings'] = embeddings

        return graph

    @staticmethod
    def visualize_graph(graph):
        # Draw the graph
        pos = nx.spring_layout(graph)
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw(graph, pos, ax=ax, with_labels=True, node_size=3000, node_color='skyblue', font_size=10,
                font_weight='bold')
        ax.set_title("Text Graph Visualization")
        plt.show()
