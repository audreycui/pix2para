from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from google_images_download import google_images_download

import sys

#Automates download of artwork images and corresponding analysis paragraph from The Smithsonian American Art Museum website

smith = 'https://americanart.si.edu'
save_dir = 'D:/download/smithsonian/'

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
    """
    prints out error
    """
    print(e)

def process(text): #removes html tags and block quote tags
	text = text.encode('ascii', errors = 'ignore')
	text = str(text)

	#text = text[text.find(':')+1:]
	ind1 = text.find('<')
	ind2 = text.find('>')
	while (ind1 >= 0 and ind2 >= 0): 
		text = text[0:ind1] + text[ind2+1:]
		ind1 = text.find('<')
		ind2 = text.find('>')

	text = remove(text, '&lt;blockquote&gt;')
	text = remove(text, '&lt;/blockquote&gt;')
	return text

def remove(text, remove): #removes all instances of 'remove' from 'text'
	ind = text.find(remove)
	while (ind>=0): 
		#print("INDEX " + str(ind) )
		text = text[0:ind] + text[ind+len(remove):]
		ind = text.find(remove)
	return text 

def processTitles(title): #removes hyphens and id numbers from titles
	title = title[title.rfind('/')+1:]
	hyphen = title.find('-')
	while(hyphen != -1): 
		title = title[0:hyphen] + ' ' + title[(hyphen+1):]
		hyphen = title.find('-')
	title = title[0:title.rfind(' ')]
	return title

def getImage(name, art_name): #automated google image search for and download of specified artwork 
	response = google_images_download.googleimagesdownload()
	args = {'keywords' : name + ' ' + art_name, 'limit': 1, 'output_directory' : save_dir}
	absolute_image_paths = response.download(args)

def iterateLetters(): 
	base_site = smith + '/art/artists'
	for i in range (143, 227): 
		#letter = chr(i+97)
		#gets html of site with all artists whose last name starts with 'letter'
		#html = BeautifulSoup(simple_get(base_site + str(i)), 'html.parser') 
		#has_next_page = html.find('li', attrs = {'class' : 'pager__item pager__item--last'})
		#counter = 0
		#while (len(has_next_page) > 0): #iterates through all pages under that letter
		if i == 0: 
			letter_site = base_site
		else:
			letter_site = base_site + '?page=' + str(i)
		letter_html = BeautifulSoup(simple_get(letter_site), 'html.parser')
		table_col = letter_html.find_all('td', attrs = {'class' : 'priority-low views-field views-field-field-alphabetical-name'})
		for artist in table_col: #iterate through all artists on that page
			artist_site = artist.find('a').get('href')
			print(artist_site + " 	page: " + str(i) )
			scrapeArtist(artist_site, processTitles(artist_site))
		#has_next_page = html.find('li', attrs = {'class' : 'pager__item pager__item--last'})
		#counter += 1

def scrapeArtist(artist_site, artist_name): 
	artist_site = smith + artist_site
	artist_html = BeautifulSoup(simple_get(artist_site), 'html.parser')
	block = artist_html.find('section', attrs = {'class' : 'views-element-container block block-views block-views-blockartists-artwork-artists-artworks-block clearfix'})
	if block is not None: 
		#print ((block))
		art_list = block.find_all('div', attrs = {'class' : 'col col-xs-12 col-sm-12 col-md-6 col-lg-3'})
		#print (len(art_list))
		for artwork in art_list: #scrape all of the artwork under that artist
			artwork_site = artwork.find('a').get('href')
			#print(artwork_site)
			scrapeArtwork(artwork_site, artist_name, processTitles(artwork_site))

def scrapeArtwork(artwork_site, artist_name, artwork_name):
	artwork_site = smith + artwork_site
	artwork_html = BeautifulSoup(simple_get(artwork_site), 'html.parser')
	caption = artwork_html.find('div', attrs = {'class' : 'field field--name-field-exhibition-label field--type-text-long field--label-above'})
	if caption is None: #scrapes exhibition label if there is one, otherwise scrapes gallery label
		caption = artwork_html.find('div', attrs = {'class' : 'field field--name-field-gallery-label field--type-text-long field--label-above'})
	if caption is not None:
		print("retrieving caption and image") 
		caption = caption.find('div', attrs = {'class' : 'field--item'})
		caption = process(caption)
		getImage(artist_name, artwork_name)
		filename = artist_name + ' ' + artwork_name
		ind_desc = open(save_dir + artist_name + ' ' + artwork_name + '/'+filename+'.txt', 'w')
		ind_desc.write(caption)
		ind_desc.close()
		#sys.exit(1)

iterateLetters()
