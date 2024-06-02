from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from CreatingGraph import TextGraphBuilder

graph_builder = TextGraphBuilder()


class GraphBasedKNN:

    def __init__(self, k=5):
        self.k = k

    def knn(self, test_graph, train_graphs):
        distances = []
        for train_graph, train_label in train_graphs:
            distance = self.graph_distance(test_graph, train_graph)
            distances.append((distance, train_label))  # Appending tuple (distance, label)
        distances.sort(key=lambda x: x[0])  # Sorting in ascending order based on distance
        # visualizing k nearest neighbors
        self.plot_distances(distances)
        nearest_neighbors = distances[:self.k]
        nearest_labels = [label for distance, label in nearest_neighbors]
        return Counter(nearest_labels).most_common(1)[0][0]  # Returning the most common label

    def plot_distances(self, distances):
        distances, labels = zip(*distances)

        # Assigning unique colors to each label
        label_color_map = {}
        unique_labels = list(set(labels))

        if len(unique_labels) == 0:
            print("Error: No unique labels found.")
            return

        # Define default colors
        default_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

        # If there are more unique labels than default colors, repeat the default colors
        colors = default_colors * ((len(unique_labels) // len(default_colors)) + 1)

        for label, color in zip(unique_labels, colors):
            label_color_map[label] = color

        # Plotting distances with labels
        plt.figure(figsize=(8, 6))
        for i, (distance, label) in enumerate(zip(distances, labels)):
            color = label_color_map[label]
            plt.scatter(distance, i, c=color)
            if i < self.k:
                plt.scatter(distance, i, facecolors='none', edgecolors='r', s=100)  # Circled points for k-nearest

        custom_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for
                         label, color in label_color_map.items()]
        custom_legend.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='r', markersize=10,
                   label='k-nearest neighbors'))
        plt.legend(handles=custom_legend)

        plt.xlabel('Distance')
        plt.ylabel('Index')
        plt.title('Distances with Labels')
        plt.show()

    @staticmethod
    def graph_distance(graph1, graph2):
        words_graph1 = set(graph1.nodes)
        words_graph2 = set(graph2.nodes)
        common_words = words_graph1.intersection(words_graph2)

        mcs = 0
        mcs_list = []
        for word in common_words:
            temp_list = []
            next_word = word
            flag = True
            length = 0
            while True:
                for words in common_words:
                    if next_word != words:
                        if graph1.has_edge(next_word, words):
                            if graph2.has_edge(next_word, words):
                                temp_list.append(next_word)
                                length += 1
                                next_word = words
                                flag = True
                                break
                    flag = False
                if not flag:
                    break
            if length == 0:
                temp_list = [word]
                length = 1
            if length > mcs:
                mcs = length
                mcs_list = temp_list

        # mcs graph visualizing
        # text = " ".join(mcs_list)
        # graph = graph_builder.build_graph_with_attributes(text)
        # graph_builder.visualize_graph(graph)

        mcs = 1 - mcs / max(len(words_graph1), len(words_graph2))
        return mcs
