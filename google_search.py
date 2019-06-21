# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:17:39 2018

@author: Benjibex
"""

#https://github.com/hardikvasa/google-images-download
#make sure you are in the google-images-download-master folder

from google_images_download import google_images_download   #importing the library, need to be in google_images-download-master folder
#from selenium import webdriver

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"Polar bears,baloons,Beaches","limit":20,"print_urls":True}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images

#arguments = {"keywords":"T-shirt,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,Ankle Boot","size":"icon","aspect_ratio":"square","format":"jpg","color_type":"black-and-white","limit":20,"print_urls":True}   #creating list of arguments
#paths = response.download(arguments)   #passing the arguments to the function
#print(paths)   #printing absolute paths of the downloaded images

arguments = {"keywords":"Pullover,Dress","format":"jpg","limit":100,"print_urls":False,"prefix":"Pullover,Dress"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images

arguments = {"keywords":"Jumper,Sweater","format":"jpg","limit":100,"print_urls":False,"prefix":"Pullover"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images

#arguments = {"keywords":"Pullover,Dress","format":"jpg","limit":10,"print_urls":True}   #creating list of arguments
#paths = response.download(arguments)   #passing the arguments to the function
#print(paths)   #printing absolute paths of the downloaded images

arguments = {"keywords":"Womens Button Shirt","prefix_keywords":"red, blue, yellow, purple, pink, floral, grey, gray, orange, green","format":"jpg","limit":100,"print_urls":False,"prefix":"Shirt"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images

arguments = {"keywords":"Dress","prefix_keywords":"red, blue, yellow, purple, pink, floral, grey, gray, orange, green","format":"jpg","limit":100,"print_urls":False,"prefix":"Dress"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images

arguments = {"keywords":"Pullover","prefix_keywords":"red, blue, yellow, purple, pink, floral, grey, gray, orange, green","format":"jpg","limit":100,"print_urls":False,"prefix":"Pullover"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images

#cant get this to work as I dont have selenium working yet
#arguments = {"keywords":"Shirt","prefix_keywords":"womens buttoned","format":"jpg","limit":200,"offset":100,"print_urls":False,"prefix":"Shirt"}   #creating list of arguments
#paths = response.download(arguments)   #passing the arguments to the function
#print(paths)   #printing absolute paths of the downloaded images

#0	T-shirt/top
#1	Trouser
#2	Pullover
#3	Dress
#4	Coat
#5	Sandal
#6	Shirt
#7	Sneaker
#8	Bag
#9	Ankle boot
