# Civit Image grabber

It downloads all the images from a provided Username, Model ID or Model TAG from CivitAI. 
Should the API not spit out all the data for all images then I'm sorry. 
The script can only download where data is provided.

The images are Downloaded into a folder with the name of the user, ModelID or the TAG <br /> 
Second Level is the Model Name with which the image was generated.


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

          Enter timeout value (in seconds): 
          Choose image quality (1 for SD, 2 for HD): 
          Choose mode (1 for username, 2 for model ID, 3 for tag search): 
          Mode 3 
          Enter tags (comma-separated): TAG
          Disable prompt check? (y/n):

                        
If you leave the timeout value emtpy it will use the default Timeout value 20 sec. <br /> 
If you leave the image quality value emtpy it will use the default image quality Value SD.

Optional: 2 or more Items which are separated with a comma



# Update History

## 0.8 Helper script
With this Script you can search locally in txt a file if your TAG is searchable. 
tagnames.py just launch it and it creates a txt File with all the Tags that the API gives out for the Model TAG search Option 3  <br />
But there are some entrys that are cleary not working. I dont kow why they are in the API Answer.  <br />
It has an function to add only new TAGS to he txt File if you run it again. 

## 0.7 Features Updates Performance 

Features: <br /> 

Model Tag Based Image download in SD or HD with Prompt Check Yes or NO <br /> 
Prompt Check YES means when the TAG is also present in the Prompt, then the image will be Downloaded. Otherwise it will be skipped.<br /> 
Prompt Check NO all Images with the searched TAG will be Downloaded. But the chance for unrelated Images is higher.<br /> 

CSV File creation within Option 3 TAG Seach  
The csv file will contain the image data that, according to the JSON file, has already been downloaded under a different TAG in this format: <br />
"Current Tag,Previously Downloaded Tag,Image Path,Download URL"  <br /> 

Litte Statistc how many images have just been downloaded and skipped with a why reasons.

Updates: <br /> 

Use of Multiple Entrys in all 3 Options comma-separated <br /> 

New Folder Structure for Downloaded Images in all Options First Folder is named after what you searched Username, ModelID, TAG. 
Second is the Model that was used to generate the image

![Untitled](https://github.com/Confuzu/CivitAI_Image_grabber/assets/133601702/fe49eb95-f1bc-4d96-80b6-c165d76d29e5)

Performance:

Code optimizations now the script runs smoother and faster. <br /> 
Better Error Handling for some Cases <br /> 


## 0.6 New Function

Rate Limiting set to 20 simultaneous connections. 
Download Date Format changend in the JSON Tracking File 


## 0.5 New Features 

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


## 0.4 Added new Functions

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

## 0.3 Added a new Function

It is writing the Meta Data for every image into a separate text file with  the ID of the image: ID_meta.txt.
If no Meta Data is available, the text file will have the URL to the image to check on the website.

Increased the timeout to 10

Added a delay between requests  
    
## 0.2 Updated with better error handling, some json validation and an option to set a timeout
