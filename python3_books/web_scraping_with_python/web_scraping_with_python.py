# -*- coding: utf-8 -*
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import re
import datetime
import random

url1 = 'http://pythonscraping.com/pages/page1.html'
url2 = 'http://www.pythonscraping.com/pages/warandpeace.html'
url3 = 'http://pythonscraping.com/pages/page3.html'

def getTitle(url):
    try:
        html = urlopen(url)
    except HTTPError as e:
        return None
    try:
        bsObj = BeautifulSoup(html.read(), features="lxml")
        title = bsObj.body.h1
    except AttributeError as e:
        return None
    return title


title = getTitle(url1)
if title == None:
    print('Title could not be found')
else:
    print(title)

html = urlopen(url2)
bsObj = BeautifulSoup(html)
namelist = bsObj.findAll('span', {'class': 'green'})
for name in namelist:
    print(name.get_text())

html = urlopen(url3)
bsObj = BeautifulSoup(html)
for child in bsObj.find('table', {'id': 'giftList'}).children:
    pass
    #print(child)

for child in bsObj.find('table', {'id': 'giftList'}).descendants:
    pass
    print(child)

for sibling in bsObj.find('table', {'id': 'giftList'}).tr.next_siblings:
    print(sibling)

images = bsObj.find_all('img', {'src': re.compile('\\.\\.\\/img\\/gifts\\/img*\\.jpg')})
for image in images:
    print(image)


url4 = 'https://en.wikipedia.org/wiki/Eric_Idle'
url5 = 'https://en.wikipedia.org/wiki/Kevin_Bacon'
html = urlopen(url4)
bsObj = BeautifulSoup(html)
for link in bsObj.find_all('a'):
    if 'href' in link.attrs:
        print(link.attrs['href'])


random.seed(datetime.datetime.now())
def getLinks(articleUrl):
    html = urlopen('https://en.wikipedia.org' + articleUrl)
    bsObj = BeautifulSoup(html)
    return bsObj.find('div', {'id': 'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))


links = getLinks('/wiki/Kevin_Bacon')
while len(links)>0:
    newArticle = links[random.randint(0, len(links)-1)].attrs['href']
    print(newArticle)
    links = getLinks(newArticle)


pages = set()
def getLinks2(pageUrl):
    global pages
    html = urlopen('https://en.wikipedia.org' + pageUrl)
    bsObj = BeautifulSoup(html)
    for link in bsObj.find_all('a', href=re.compile('^(/wiki/)')):
        if link.attrs['href'] not in pages:
            newPage = link.attrs['href']
            print(newPage)
            pages.add(newPage)
            getLinks2(newPage)
