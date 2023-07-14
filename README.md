# Civit Image grabber

It grabs all the uploaded images from a provided Username from Civit.  

# Usage 
```
pip install -r requirements.txt
```
```
python civit_image_downloader.py  
```
The script  will ask you for a username and a timeout value.

If you leave the timeout value emtpy it will use the Default timeout value 5 sec.

Optional: 2 or more Usernames which are separated with a comma

but the more usernames the more connections and api calls results in more Failed connection. 
