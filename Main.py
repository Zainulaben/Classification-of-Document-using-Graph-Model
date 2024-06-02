import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from CreatingGraph import TextGraphBuilder
from KNNClassifier import GraphBasedKNN
from TextPreprocessing import TextPreprocessor
from WebScrapping import WebsiteScraper
from sklearn.metrics import classification_report

# Web Scraping
scraper = WebsiteScraper()
preprocessor = TextPreprocessor()
graph_builder = TextGraphBuilder()

directory_names = ['Web Data/Sports/Training', 'Web Data/Sports/Testing', 'Web Data/Diseases & Symptoms/Training',
                   'Web Data/Diseases & Symptoms/Testing',
                   'Web Data/Lifestyle & Hobbies/Training', 'Web Data/Lifestyle & Hobbies/Testing']

testing_directories = ['Web Data/Sports/Testing', 'Web Data/Diseases & Symptoms/Testing',
                       'Web Data/Lifestyle & Hobbies/Testing']

# Sports
s_training_urls = ['https://en.wikipedia.org/wiki/Sport', 'https://en.wikipedia.org/wiki/Racing',
                   'https://en.wikipedia.org/wiki/Volleyball', 'https://en.wikipedia.org/wiki/Table_tennis',
                   'https://en.wikipedia.org/wiki/Basketball', 'https://en.wikipedia.org/wiki/Baseball',
                   'https://en.wikipedia.org/wiki/Golf', 'https://en.wikipedia.org/wiki/Rugby_football',
                   'https://en.wikipedia.org/wiki/Association_football', 'https://en.wikipedia.org/wiki/Cricket',
                   'https://en.wikipedia.org/wiki/Hockey', 'https://en.wikipedia.org/wiki/Tennis']

s_testing_urls = ['https://en.wikipedia.org/wiki/Motorsport', 'https://en.wikipedia.org/wiki/International_sport',
                  'https://en.wikipedia.org/wiki/Sports_film']

# Diseases & Symptoms
d_training_urls = ['https://en.wikipedia.org/wiki/Disease', 'https://en.wikipedia.org/wiki/Signs_and_symptoms',
                   'https://en.wikipedia.org/wiki/Acne', 'https://en.wikipedia.org/wiki/Cardiovascular_disease',
                   'https://en.wikipedia.org/wiki/Cancer', 'https://en.wikipedia.org/wiki/Stroke',
                   'https://en.wikipedia.org/wiki/Respiratory_disease', 'https://en.wikipedia.org/wiki/Diabetes',
                   'https://en.wikipedia.org/wiki/Alzheimer\'s_disease', 'https://en.wikipedia.org/wiki/Allergy',
                   'https://en.wikipedia.org/wiki/Asthma', 'https://en.wikipedia.org/wiki/Gastrointestinal_disease']

d_testing_urls = ['https://en.wikipedia.org/wiki/Bronchitis', 'https://en.wikipedia.org/wiki/Malaria',
                  'https://en.wikipedia.org/wiki/Dengue_fever']

# Lifestyle & Hobbies
l_training_urls = ['https://vocal.media/unbalanced/hobby-vs-lifestyle', 'https://en.wikipedia.org/wiki/Hobby_farm',
                   'https://en.wikipedia.org/wiki/Hobby', 'https://www.dermatologytimes.com/view/tailoring-ad-management-advice-to-lifestyles-and-hobbies',
                   'https://www.lifestyledaily.co.uk/article/2020/03/26/ten-lifestyle-hobbies-take-during-lockdown-keep-yourself-busy', 'https://www.mdpi.com/1660-4601/13/1/135',
                   'https://online.kettering.edu/news/why-hobbies-are-important', 'https://info.totalwellnesshealth.com/blog/11-healthy-hobbies-you-can-start-today',
                   'https://itsocoffee.com/blogs/news/how-is-coffee-linked-to-our-daily-lifestyle-and-hobbies-health-benefits-of-drinking-coffee', 'https://www.huffpost.com/entry/healthy-hobbies-that-will-improve-your-life_b_589a17c8e4b0985224db5ab6',
                   'https://en.wikipedia.org/wiki/Lifestyle_(social_sciences)', 'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0255359']

l_testing_urls = ['https://www.netmeds.com/health-library/post/get-a-hobby-for-a-happier-healthier-lifestyle', 'https://sometimes-homemade.com/lifestyle-hobbies',
                  'https://thehobbie.com/lifestyle']

urls = [s_training_urls, s_testing_urls, d_training_urls, d_testing_urls, l_training_urls, l_testing_urls]
graph_list = []
testing_graphs = []
testing_labels = []

# # web scrapping
for url, directory in zip(urls, directory_names):
    scraped_data = scraper.scrape_websites(url)
    preprocessed_data = [preprocessor.preprocess_text(value) for value in scraped_data.values()]
    scraper.save_to_files(preprocessed_data, url, directory)

# graph visualization
data = ""
for directory in directory_names:
    label = directory.split('/')[-2]
    directory_type = directory.split('/')[-1]
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = ""
                data += file.read()
                graph_with_attributes = graph_builder.build_graph_with_attributes(data)
                if directory_type == 'Testing':
                    testing_graphs.append(graph_with_attributes)
                    testing_labels.append(label)
                else:
                    graph_list.append((graph_with_attributes, label))

# Instantiate GraphBasedKNN
graph_knn = GraphBasedKNN(k=3)

# Perform KNN classification on testing graphs
predicted_labels = []
for test_graph in testing_graphs:
    predicted_label = graph_knn.knn(test_graph, graph_list)
    predicted_labels.append(predicted_label)

# Print predicted labels
print("Predicted labels:", predicted_labels)

# Calculate confusion matrix
conf_matrix = confusion_matrix(testing_labels, predicted_labels, labels=sorted(set(testing_labels)))
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix using seaborn and matplotlib
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=sorted(set(testing_labels)), yticklabels=sorted(set(testing_labels)))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Calculate accuracy, precision, recall, and F1 score
report = classification_report(testing_labels, predicted_labels, target_names=sorted(set(testing_labels)))
print("Classification Report:")
print(report)

# Calculate accuracy
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
print("Accuracy:", accuracy)
