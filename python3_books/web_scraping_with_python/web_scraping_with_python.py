# -*- coding: utf-8 -*
from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import re
import datetime
import random
import os
import csv

url1 = 'http://pythonscraping.com/pages/page1.html'
url2 = 'http://www.pythonscraping.com/pages/warandpeace.html'
url3 = 'http://pythonscraping.com/pages/page3.html'
url4 = 'http://pythonscraping.com'

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

# 下载单张图片
html = urlopen(url4)
bsObj = BeautifulSoup(html)
imageLocation = bsObj.find('a', {'id': 'logo'}).find('img')['src']
urlretrieve(imageLocation, 'logo.jpg')


downloadDirectory = 'download'
baseUrl = 'http://pythonscraping.com'


def get_absolute_url(baseUrl, source):
    if source.startswith('https://www.'):
        url = 'http://' + source[11:]
    elif source.startswith('http://'):
        url = source
    elif source.startswith('www.'):
        url = source[4:]
        url = 'http://' + url
    else:
        url = baseUrl + '/' + source
    if baseUrl not in url:
        return None
    return url


def get_download_path(baseUrl, absolute_url, download_directory):
    path = absolute_url.replace('www.', '')
    path = path.replace(baseUrl, '')
    path = download_directory + path
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.mkdir(directory)
    return path


html = urlopen(baseUrl)
bsObj = BeautifulSoup(html)
downloadList = bsObj.find_all(src=True)

for download in downloadList:
    fileUrl = get_absolute_url(baseUrl, download['src'])
    if fileUrl is not None:
        print(fileUrl)

#urlretrieve(fileUrl, get_download_path(baseUrl, fileUrl, downloadDirectory))

# 下载html表格并存储在csv文件中
url5 = 'http://en.wikipedia.org/wiki/Comparison_of_text_editors'
html = urlopen(url5)
bsObj = BeautifulSoup(html)
table = bsObj.find_all('table', {'class': 'wikitable'})[0]
rows = table.find_all('tr')

csvFile = open('editors.csv', 'wt', newline='', encoding='utf-8')
writer = csv.writer(csvFile)
try:
    for row in rows:
        csvRow = []
        for cell in row.find_all(['td', 'th']):
            csvRow.append(cell.get_text())
            writer.writerow(csvRow)
finally:
    csvFile.close()


