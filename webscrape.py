import requests
from bs4 import BeautifulSoup
import json
import nltk
from nltk.tokenize import word_tokenize

def crawl_months_links(sitemap_url):
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.content, "html.parser")

    year_tags = soup.find_all('div', class_='sc-zpw6hx-1 dPzjQj')

    months_articles_links = []
    for year_tag in year_tags:
        month_links = year_tag.find_all('a', class_='js_sitemap-month')
        for month_link in month_links:
            month_url = month_link['href']
            # replace exisitng /sitemap with new /sitemap link
            month_url = month_url.replace('/sitemap', '')
            absolute_url = sitemap_url + month_url
            months_articles_links.append(absolute_url)

    return months_articles_links

def crawl_articles(months_links):
    #print(months_links)
    articles = []
    for link in months_links:
        #print('link:', link)
        response = requests.get(link)
        #print('response:',response)
        soup = BeautifulSoup(response.content, "html.parser")
        #print('soup:',soup)

        article_tags = soup.find_all('h4', class_='sc-1w8kdgf-1 lbfjcR js_sitemap-article')
        #print('article tags:', article_tags)
        for article_tag in article_tags:
            article_url = article_tag.find('a')['href']
            #print('article_url:', article_url)
            articles.append({
                'url': article_url,
            })

    return articles

def crawl_article(article_url):
    response = requests.get(article_url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    #print("Article URL:", article_url)
    
    title_elem = soup.find('h1', class_='sc-1efpnfq-0 dAlcTj')
    if title_elem is not None:
        title = title_elem.text
    else:
        title = ""
        
    #print("Title:", title)

    content_elem = soup.find('p', class_='sc-77igqf-0 fnnahv')
    if content_elem is not None:
        content = content_elem.text
    else:
        content = ""
        
    #print("Content:", content)

    dict = {'title': title, 'content': content}
    print(dict)
    return dict

sitemap_url = 'https://www.theonion.com/sitemap'

# links for months during which publications were made
months_links = crawl_months_links(sitemap_url)

# links of articles during any active month in year
articles = crawl_articles(months_links)

articles_contents = []
articles_contents_tokenized = []

for link in articles:
    dict = crawl_article(link['url'])
    articles_contents.append(dict)

for article in articles_contents:
    title = article['title']
    content = article['content']
    title_tokens = word_tokenize(title)
    content_tokens = word_tokenize(content)
    articles_contents_tokenized.append({
        'title_tokens': title_tokens,
        'content_tokens': content_tokens
    })

with open('articles.json', 'w') as outfile:
    json.dump(articles_contents_tokenized, outfile)
