# Civit Image grabber

It downloads all the images from a provided Username or Model ID from CivitAI. 
Should the API not spit out all the data for all images then I'm sorry. 
The script can only download where data is provided.

The images are Downloaded into a folder with the name of the user or the ModelID.


# CivitAI API is fixed

# Usage 
```
install Python3
```
```
pip install -r requirements.txt
```
```
python civit_image_downloader.py
```
The script  will ask you to 

                        Choose mode (1 for username, 2 for model ID): 
                        timeout value (in seconds):
                        image quality (1 for SD, 2 for HD):
                        usernames (comma-separated):

If you leave the timeout value emtpy it will use the default Timeout value 20 sec.

Optional: 2 or more Usernames which are separated with a comma

but the more usernames the more connections and api calls results in more Failed connection. 



# Update History


0.6 New Function

Rate Limiting set to 20 simultaneous connections. 
Download Date Format changend in the JSON Tracking File 


0.5 New Features 

Option for Downloading SD (jpeg) Low Quality or HD (PNG) High Quality Version of Images


Better Tracking of images that already downloaded, with a JSON File called downloaded_images.json in the same Folder as the script. The Scripts writes 
for SD Images with jpeg Ending
```
        "ImageID_SD": 
        "path": "image_downloads/civitAIuser/image.jpeg",
        "quality": "SD",
        "download_date": "YYYY-MM-DD - H:M"       
```
For HD Images with PNG Ending
```
        "ImageID_HD": {
        "path": "image_downloads/civitAIuser/Image.png",
        "quality": "HD",
        "download_date": "YYYY-MM-DD- H:M"
```
into it and checks before Downloading a Image. For Both Option, Model ID or Username


0.4 Added new Functions

Image Download with Model ID. Idea for it came from bbdbby 
The outcome looks sometimes chaotic a lot of folders with Modelnames you cant find on CivitAI. 
Because of renaming or Deleting the Models. But older Images have the old Model Names in the META data. 


Sort function to put the images and meta txt files into the right Model Folder. 
The sort Function relies on the Meta Data from the API for the images. Sometimes Chaos. 
Especially for models that have a lot of images.


Tracking of images that already downloaded with a text file called downloaded_images.txt in the same Folder as the script.
The Scripts writes the Image ID into  it and checks before Downloading a Image. 
For Both Option, Model ID or Username

Increased the timeout to 20

0.3 Added a new Function

It is writing the Meta Data for every image into a separate text file with  the ID of the image: ID_meta.txt.
If no Meta Data is available, the text file will have the URL to the image to check on the website.

Increased the timeout to 10

Added a delay between requests  
    
0.2 Updated with better error handling, some json validation and an option to set a timeout
