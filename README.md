# Civit Image grabber

It grabs all the uploaded images from a provided Username from Civit. Over the civitAI API 
Should the api not spit out all the data for all images then I'm sorry 
The script can only download where data is provided.


# Usage 
```
pip install -r requirements.txt
```
```
python civit_image_downloader.py  
```
The script  will ask you for a username and a timeout value.

If you leave the timeout value emtpy it will use the default Timeout value 5 sec.

Optional: 2 or more Usernames which are separated with a comma

but the more usernames the more connections and api calls results in more Failed connection. 



# Update History

0.3 Added a new Function 

It is writing the Meta Data for every image into a separate text file with  the ID of the image: ID_meta.txt.
If no Meta Data is available, the text file will have the URL to the image to check on the website.

Increased the timeout 

Added a delay between requests  
    
0.2 Updated with better error handling, some json validation and an option to set a timeout
