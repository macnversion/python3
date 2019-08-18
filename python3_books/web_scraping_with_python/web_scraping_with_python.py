# -*- coding: utf-8 -*
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup

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