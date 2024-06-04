"""
Dataset:
@phdthesis{MnihThesis,
    author = {Volodymyr Mnih},
    title = {Machine Learning for Aerial Image Labeling},
    school = {University of Toronto},
    year = {2013}
}
"""

import os
import argparse
import requests
from bs4 import BeautifulSoup

from .config import config


def get_content(url):
    return requests.get(url).content


counter = 0


def download_url_to(url, path):
    global counter
    response = requests.get(url)
    os.chdir(path)

    with open(str(counter) + '.tiff', 'wb') as f:
        print("Write "+f.name+" to "+os.path.abspath(f.name))
        f.write(response.content)

    counter += 1


def get_tiff_links_from_page(url):
    soup = BeautifulSoup(get_content(url), 'html.parser')

    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith('.tiff') or href.endswith('.tif'):
            links.append(href)
    return links


def main():
    parser = argparse.ArgumentParser(description='Download dataset for road segmentation')
    parser.add_argument('save_path', type=str, help='Path on current host where to save dataset')
    args = parser.parse_args()

    for sample_type, url in config['urls'].items():
        sample_path = args.save_path + "/" + sample_type
        os.makedirs(sample_path, 0x755, exist_ok=True)
        for u in get_tiff_links_from_page(url):
            download_url_to(u, sample_path)


if __name__ == '__main__':
    main()
