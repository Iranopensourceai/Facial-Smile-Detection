##### Data visualization (method 1) 

#import dependencies 
import tensorflow as tf 
import matplotlib.pyplot as plt
import os 
#from data_visualization import visualization

img_path = 'Datasetpath/*.jpg'                                         #defining Dataset path 

def data_visualization(img_path,n_cols,batch_size):                    #function to visualize images in dataset path
    images = tf.data.Dataset.list_files(img_path)                      #Load RAW images into tf data pipeline
    images.as_numpy_iterator().next()
    def load_image(x):                                                 #function to read and decode images 
        byte_img = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byte_img)
        return img
    images = images.map(load_image)                                    #mapping load image Function to our dataset images
    images.as_numpy_iterator().next() 
    image_generator = images.batch(batch_size).as_numpy_iterator()     #go to the next batch of images
    plot_images = image_generator.next()

    #plotting RAW image
    fig, ax = plt.subplots(ncols= n_cols, figsize=(30,30))
    for idx, image in enumerate(plot_images): 
        ax[idx].imshow(image)
        
    plt.show() #showing plot

data_visualization('Datasetpath/*.jpg',3,3) #calling data_visualization(images_path,number of columns,batchsize)
)


##### Data visualization (method 2)

#import dependencies
import cv2
from matplotlib import pyplot as plt
import os


def visualization(n_pic,folder,ncols): #(number of images,path,number of columns)
    images = []
    folder_dir = folder
    #open file and select every items 
    for img in os.listdir(folder_dir): 
 
    # check if the image ends with jpg or png if u need png u can add it here
        if (img.endswith(".jpg")):
            images.append(img)
    for x in range (0,n_pic):
        a = folder +"/"+ images[x] #for complete address text
        y = x+1
        nrows = int(n_pic/ncols+1)
        pic = cv2.imread(a)
        
        plt.subplot(nrows,ncols,y),plt.imshow(pic)
        plt.title(images[x]), plt.xticks([]), plt.yticks([]) 
    plt.show()
visualization(12,"E:/smile project/data/smile/",4)
        
 
