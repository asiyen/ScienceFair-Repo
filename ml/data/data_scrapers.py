import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import os
def csv_read_url(file):
  ldf=list(str(file).split('\n'))
  types=[[] for i in ldf[0].split(',')]
  for i in ldf[1:]:
    split=i.split(',')
    for j in range(len(types)):
      types[j].append(split[j])
  return types  
def messy_read(file,truth=1):
  with open('a.csv','w') as f:
    f.write(str(file))
  read=pd.read_csv('a.csv')
  read.insert(0,'label',[truth]*len(read),True)
  os.remove('a.csv')
  return read
def go_to_git(url):
  r=requests.get(url)
  soup=bs(r.content)
  dfs=[]
  for i in soup.find_all('a',{'class':'js-navigation-open link-gray-dark'}):
    print('i')
    #gets each of the folders
    dfs+=get_raw('https://github.com'+i['href'])
  return dfs
git_start='https://github.com'
def get_raw(url):
  r=requests.get(url)
  soup=bs(r.content)
  dfs=[]
  for t in soup.find_all('a',{'class':'js-navigation-open link-gray-dark'}):
    soup2=bs(requests.get('https://github.com'+t['href']).content)
    for tag in soup2.find_all('a',{'id':'raw-url'}):
      dfs.append(messy_read(requests.get(git_start+tag['href']).text,'Real' in t['title']))
  return dfs
all_dfs=go_to_git('https://github.com/cuilimeng/CoAID')
