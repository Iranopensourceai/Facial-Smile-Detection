# Importing necessary libraries
import os
import cv2
import time
import glob
import urllib
import pathlib
import argparse
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.firefox import GeckoDriverManager


# argparse
parser = argparse.ArgumentParser(description='Process command line arguments.')
parser.add_argument("u", help="urls path", type=pathlib.Path)
parser.add_argument("c", help="cascade path", type=pathlib.Path)
args = parser.parse_args()



def get_Sources(urls_path):
    # urls for perform searches in Google pages using selenium
    urls = pd.read_csv(urls_path)

    Smile_src = []
    nonSmile_src = []

    for i in range(len(urls)):
        url = urls.iloc[i]

        # WebDriver is a remote control interface that enables introspection and control of user agents.
        # ChromeDriver is a standalone server that implements the W3C WebDriver standard.
        driver = webdriver.Firefox(GeckoDriverManager().install())

        # get() is used to navigate particular URL(website) and wait till page load
        driver.get(url.values[0])

        # Scroll down webpage
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # Find Elements By Class name in Selenium WebDriver    
        imgResults = driver.find_elements(By.CLASS_NAME,"Q4LuWd")

        # append image sources to list
        for img in imgResults:
            if img.get_attribute("src") and "http" in img.get_attribute('src'):
                if url.name == 'Smile':
                    Smile_src.append(img.get_attribute('src'))
                elif url.name == 'nonSmile':
                    nonSmile_src.append(img.get_attribute('src'))

        # close the current browser window
        driver.close()

    # smile DataFrame
    df_Smile = pd.DataFrame(Smile_src)
    df_Smile.set_index([['Smile']*len(df_Smile)], inplace=True)

    # non smile DataFrame
    df_nonSmile = pd.DataFrame(nonSmile_src)
    df_nonSmile.set_index([['nonSmile']*len(df_nonSmile)], inplace=True)

    # concat
    df = pd.concat([df_Smile, df_nonSmile], axis=0)

    # drop duplicates
    df = df.iloc[:,0:1].drop_duplicates()
    df.to_csv('image_Sources.csv', columns=['image_Sources'])





def download_images(sources_path='image_Sources.csv'):
    df = pd.DataFrame(sources_path)

    # Download and save images to a folder
    os.mkdir('Data')
    os.mkdir('Data/1')
    os.mkdir('Data/0')

    for j in range(len(df)):
        src = df.iloc[j]

        if src.name == 'Smile':
            urllib.request.urlretrieve(str(src.values[0]),"Data/1/Smile{}.jpg".format(j))
        elif src.name == 'nonSmile':
            urllib.request.urlretrieve(str(src.values[0]),"Data/0/nonSmile{}.jpg".format(j))

    


def data_cleaning(cascade_path):
    # Data Cleaning and save images to a folder
    os.mkdir('Dataset')
    os.mkdir('Dataset/1')
    os.mkdir('Dataset/0')

    # Load the cascade for Face Detection
    face_cascade = cv2.CascadeClassifier(cascade_path)

    for address in glob.glob('Data/*/*'):

        img = cv2.imread(address)

        # cascade works with grayscale images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x-30, y-30), (x+w +30, y+h +30), (255, 0, 0), 2)

        # crop faces from images
        crop_img = img[y-10: y+h+10, x-10: x+w+10]

        if crop_img.shape[0] > 20 and crop_img.shape[1] > 20:

            d = address.split('\\')
            path = f'Dataset/{d[-2]}/'
            cv2.imwrite(os.path.join(path , d[-1]), crop_img)




if __name__ == "__main__" :
    # The src attribute specifies the URL or path of the image to be displayed. we can download images from it.
    get_Sources(args.u)

    # Download images
    download_images()

    # Cropping Faces from Images for images classification
    data_cleaning(args.c)
