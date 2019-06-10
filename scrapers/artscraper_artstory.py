from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from google_images_download import google_images_download
import re
import urllib
import sys 
from imp import reload 

#Automates download of artwork images and corresponding analysis paragraph from The Art Story 

def simple_get(url):
    """
    gets html content at url
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None

def is_good_response(resp):
    """
    returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)


def log_error(e):
	#prints out error
    print(e)

def getDescAndImage(artist_html, all_desc):

	#print(artist_html)
	artist_html = simple_get(artist_html)
	#print(artist_html)
	artist_html = BeautifulSoup(artist_html, 'html.parser')

	#artist's name
	name = artist_html.find('h1', attrs = {'class' : 'name'})
	#print (name)
	name = str(name)
	name = name[name.find('>')+1 : name.find('</')]

	#list of all artwork
	art_names = artist_html.find_all('h3', attrs = {'class' : 'artwork-title'})

	#list of all artwork descriptions
	artwork_desc = artist_html.find_all('p', attrs = {'class' : 'artwork-desc'})

	for i in range(0, len(art_names)):
		art_names[i] = art_names[i].encode('ascii', errors = 'ignore')
		art_names[i] = str(art_names[i])
		art_names[i] = art_names[i][art_names[i].find('>')+1 : art_names[i].find('</')]
		art_names[i] = removeCommas(art_names[i])
		name = removeCommas(name)
		print(name)
		print (art_names[i])
		art_name = art_names[i]
		if len(art_names[i])>50:
			art_name = art_name[0:50]
		art_name = art_name.strip()

		try:
			getImage(name, art_name)
		except Exception:
			print("failed: " + name + str(art_name))
			continue

		d = process(artwork_desc[i])
		#all_desc.write(d)
		filename = name + ' ' + art_name
		ind_desc = open('D:/download/artstory_redo/'+name + ' ' + art_name+ '/'+filename+'.txt', 'w')
		ind_desc.write(d)
		ind_desc.close()
		

		

def removeCommas (title):
	
	ind = title.find(',')

	while (ind >= 0):
		title = title[0:ind] + title[ind+1:]
		ind = title.find(',')

	ind = title.find('/')

	while (ind >= 0):
		title = title[0:ind] + title[ind+1:]
		ind = title.find('/')

	ind = title.find('Important Art')
	
	if (ind>=0):
		title = title[0:ind] + title[ind+len('Important Art'):]

	ind = title.find('\\')
	
	while (ind>=0):
		title = title[0:ind] + title[ind+1:]
		ind = title.find('\\')

	return title

def getImage(name, art_name): 
	
	response = google_images_download.googleimagesdownload()
	
	args = {'keywords' : name + ' ' + art_name, 'limit': 1, 'output_directory' : 'D:/download/artstory_redo/'}
	absolute_image_paths = response.download(args)


def process(text):
	text = text.encode('ascii', errors = 'ignore')
	text = str(text)

	text = text[text.find(':')+1:]
	ind1 = text.find('<')
	ind2 = text.find('>')

	while (ind1 >= 0 and ind2 >= 0): 
		text = text[0:ind1] + text[ind2+1:]
		ind1 = text.find('<')
		ind2 = text.find('>')
	return (text)

def iterateLetters():

	#all_desc = open('/Users/audreycui01/Documents/artanalysis/alldescriptions.txt', 'a')

	arthtml = ('D:/download/artstory_web.htm')
	#html = simple_get(arthtml)
	html = open(arthtml, encoding='utf-8')

	html = BeautifulSoup(html, 'html.parser')
	artist_block = html.find_all('div', attrs = {'class' : 'artist-block'})
	print (len(artist_block))
	#for i in range (0, len(artist_block)):
	for i in range (430, len(artist_block)):
		raw_artist_html = artist_block[i].find_next('a').get('href')
		raw_artist_html = raw_artist_html[0:(raw_artist_html.find('.htm'))]
		artist_html = raw_artist_html + '-artworks.htm'
		print (i)
		print (artist_html)
		
		getDescAndImage(artist_html, None)
		
	#print (desc)
	#all_desc.close()

iterateLetters()
