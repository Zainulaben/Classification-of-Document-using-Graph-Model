import requests
from bs4 import BeautifulSoup
import os


class WebsiteScraper:
    def __init__(self):
        pass

    @staticmethod
    def scrape_websites(urls):
        scraped_data = {}
        for url in urls:
            # Send a GET request to the URL
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the HTML content of the page
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract text from paragraphs
                paragraphs = soup.find_all('p')
                text = ' '.join([p.get_text() for p in paragraphs])

                # Store the scraped text in a dictionary
                scraped_data[url] = text
            else:
                print("Failed to retrieve content from the URL:", url)

        return scraped_data

    @staticmethod
    def save_to_files(data, urls, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        for url, text in zip(urls, data):
            filename = os.path.join(directory, url.split('/')[-1] + ".txt")
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(text)
