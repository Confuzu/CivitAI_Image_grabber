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
## Interactive Mode
the script will ask you to:

          Enter timeout value (in seconds) [default: 60]: 
          Choose image quality (1 for SD, 2 for HD) [default: 1]: 
          Allow re-downloading of images already tracked (1 for Yes, 2 for No) [default: 2]: 
          Choose mode (1 for username, 2 for model ID, 3 for Model tag search, 4 for model version ID):
          Mode 3 
          Enter tags (comma-separated): TAG
          Disable prompt check? (y/n):

                        
If you just hit enter it will use the Default values of that Option if it has a default value.  <br /> 
 <br /> 
## Command-Line Mode
arguments filled with the default values as an example<br /> 
Username Mode
```
--timeout=60  --quality=1 --redownload=2  --mode=1   --username=
```
Model ID Mode
```
--timeout=60  --quality=1 --redownload=2  --mode=2   --model_id=
```
Tag search Mode 
```
--timeout=60  --quality=1 --redownload=2  --mode=3   --tags=  --disable_prompt_check=
```
Model version ID Mode
```
--timeout=60  --quality=1 --redownload=2  --mode=4   --model_version_id=
```
 <br /> 
 <br /> 
 
## Mixed Mode 
If only some arguments are provided, the script will use the provided options and prompt the user for any missing inputs. <br /> 
 <br /> 



## Folder Structure  <br /> 
The downloaded files will be organized in the following structure:
```
image_downloads/
└── Username_Search/
|   ├── Username/
│       ├── Model1/
│       |   ├── image1.jpeg
│       |   ├── image1.png
│       |   └── details.txt
│       ├── Model2/
│       |   ├── image1.jpeg
│       |   ├── image1.png
│       |   └── details.txt
│       ├── invalid_meta/
│       |   ├── image1.jpeg
│       |   ├── image1.png
│       |   └── details.txt
│       ├── no_meta_data/
│           ├── image1.jpeg
│           ├── image1.png
│           └── details.txt

├── Model_ID_Search/
│   └── Model_ID/
│       ├── Model1/
│       |   ├── image1.jpeg
│       |   ├── image1.png
│       |   └── details.txt
│       ├── Model2/
│       |   ├── image1.jpeg
│       |   ├── image1.png
│       |   └── details.txt
│       ├── invalid_meta/
│       |   ├── image1.jpeg
│       |   ├── image1.png
│       |   └── details.txt
│       ├── no_meta_data/
│           ├── image1.jpeg
│           ├── image1.png
│           └── details.txt

├── Model_Version_ID_Search/
│   └── Version_ID/
│       ├── Model1/
│       |   ├── image1.jpeg
│       |   ├── image1.png
│       |   └── details.txt
│       ├── Model2/
│       |   ├── image1.jpeg
│       |   ├── image1.png
│       |   └── details.txt
│       ├── invalid_meta/
│       |   ├── image1.jpeg
│       |   ├── image1.png
│       |   └── details.txt
│       ├── no_meta_data/
│           ├── image1.jpeg
│           ├── image1.png
│           └── details.txt

├── Model_Tag_Search/
│   └── Searched_tag/
│       ├── model_ID/
│           ├──Model1/
│           |   ├── image1.jpeg
│           |   ├── image1.png
│           |   └── details.txt
│           └── Searched_tag_summary_YYYYMMDD.csv
│           ├──Model2/
│           |   ├── image1.jpeg
│           |   ├── image1.png
│           |   └── details.txt
│           └── Searched_tag_summary_YYYYMMDD.csv
│           ├──invalid_meta/
│           |   ├── image1.jpeg
│           |   ├── image1.png
│           |   └── details.txt
│           └── Searched_tag_summary_YYYYMMDD.csv
│           ├──no_meta_data/
│           |   ├── image1.jpeg
│           |   ├── image1.png
│           |   └── details.txt
│           └── Searched_tag_summary_YYYYMMDD.csv
```
# Update History

## 1.2 New Feature & Update

### Command-Line Parameter Support <br />

This update introduces support for three different startup modes.<br />

Fully Interactive Mode: If no command-line arguments are provided, the script will prompt the user for all required inputs interactively, as before.<br />

Fully Command-Line Mode: If all necessary arguments are supplied via the command line, the script will execute without any prompts, offering a streamlined experience for advanced users.<br />

Mixed Mode: If only some arguments are provided, the script will use the provided options and prompt the user for any missing inputs. This allows for a flexible combination of both modes.<br />

The new Feature includes a check for mismatched arguments. If you provide arguments that don't match the selected mode, you will receive a warning message, but the script will continue to run,<br /> 
ignoring the mismatched arguments and prompting for the required information if necessary.<br />

```
Warning: --Argument is not used in ... mode. This argument will be ignored.
```
## no_meta_data Folder
All images with no_meta_data are now moved to their own folder named no_meta_data. <br />
They also have a text file containing the URL of the image, rather than any metadata.<br />
```
No metadata available for this image.
URL: https://civitai.com/images/ID?period=AllTime&periodMode=published&sort=Newest&view=feed&username=Username&withTags=false
```


### Update
## BUG FIX
A bug was fixed where the script sometimes did not download all the images provided by the API.<br />
The logging function was also enhanced. You can now see how many image links the API provided and what the script has downloaded. <br />
A short version is displayed in your terminal. <br />
```
Number of downloaded images: 2
Number of skipped images: 0
```
While more detailed information is available in the log file.<br />
```
Date Time - INFO - Running in interactive mode
Date Time - WARNING - Invalid timeout value. Using default value of 60 seconds.
Date Time - WARNING - Invalid quality choice. Using default quality SD.
Date Time - INFO - Received 2 items from API for username Example
Date Time - INFO - Attempting to download: https://image.civitai.com/247f/width=896/b7354672247f.jpeg
Date Time - INFO - Attempting to download: https://image.civitai.com/db84/width=1024/45757467b84.jpeg
Date Time - INFO - Successfully downloaded: image_downloads/Username_Search/Example/2108516.jpeg
Date Time - INFO - Successfully downloaded: image_downloads/Username_Search/Example/2116132.jpeg
Date Time - INFO - Marked as downloaded: 21808516 at image_downloads/Username_Search/Example/2108516.jpeg
Date Time - INFO - Marked as downloaded: 21516132 at image_downloads/Username_Search/Example/2116132.jpeg
Date Time - INFO - Total items from API: 2, Total downloaded: 2
Date Time - INFO - 2 images have no meta data.
Date Time - INFO - Total API items: 2, Total downloaded: 2
Date Time - INFO - Image download completed.

```





## 1.1 New Feature & Update

### New Download Option Modelversion ID   <br />
The script can now selectively download images that belong to a specific model version ID. Option 4 <br />
This saves disk space and in addition, the Civit AI Server API is used less, which leads to a more efficient use of resources. <br />
The Script will download the Images to this new Folder  --> Model_Version_ID_Search<br />
Updated the **Folder Structure** <br />


### Updated Timeout  <br />
i have noticed that the timeout of 20 seconds is too short for model ID and version ID and that i get more network errors than downloads,  <br />
so i have set it to 60 seconds for now.  <br />
But if you want to be on the safe side, then enter the following: 120  for the option: Enter timeout value (in seconds): <br />
this has always worked up to now <br />


## 1.0  Update

Updated Folder Structure. <br />
The script creates a Folder for each Option you can choose.  <br />
This new structure ensures better organization based on the search type, making image management more efficient. <br />

## 0.9 Feature & Updates

New Feature

Redownload of images.
The new option allows the tracking file to be switched off. So that already downloaded images can be downloaded again. 
```
Allow re-downloading of images already tracked (1 for Yes, 2 for No) [default: 2]: 
```
If you choose 2 or just hit enter the Script will run with Tracking as Default like always. <br />


New Update <br />

When the script is finished, a summary of the usernames or Model IDs that could not be found is displayed. <br />
```
Failed identifiers:
username: 19wer244rew
```
```
Failed identifiers:
ModelID: 493533
```


## 0.8 Helper script tagnames
With this Script you can search locally in txt a file if your TAG is searchable.  <br />
Just launch tagnames.py and it creates a txt File with all the Tags that the API gives out for the Model TAG search Option 3  <br />
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
