#!/usr/bin/env python
import httpx
import os
import sys
import asyncio
import aiofiles 
import json
import time
from tqdm import tqdm
import shutil
import re
from datetime import datetime
import logging
import csv
from threading import Lock
import argparse

##########################################
# Helper: Detect file type from its signature
##########################################
def detect_extension(data: bytes) -> str:
    """
    Detects the file extension from the first bytes of the data.
    Supports: PNG, JPEG, WebP, MP4 and WebM.
    """
    # Check for PNG: 89 50 4E 47 0D 0A 1A 0A
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return ".png"
    # Check for JPEG: FF D8 FF
    if data.startswith(b'\xff\xd8\xff'):
        return ".jpeg"
    # Check for WebP: starts with 'RIFF' and then has 'WEBP' at bytes 8-12
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return ".webp"
    # Check for MP4: many MP4 files contain 'ftyp' at byte offset 4
    if len(data) >= 12 and data[4:8] == b'ftyp':
        return ".mp4"
    # Check for WEBM: EBML header (commonly used by WebM)
    if data.startswith(b'\x1A\x45\xDF\xA3'):
        return ".webm"
    return None

##########################################
# Setup logging
##########################################
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "civit_image_downloader_log_1.3.txt")
logger_cid = logging.getLogger('cid')
logger_cid.setLevel(logging.DEBUG)
file_handler_cid = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler_cid.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler_cid.setFormatter(formatter)
logger_cid.addHandler(file_handler_cid)

##########################################
# CivitAi API is fixed!
# civit_image_downloader_1.3
##########################################

# API endpoint for retrieving image URLs
base_url = "https://civitai.com/api/v1/images"

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Content-Type": "application/json"
}

semaphore = asyncio.Semaphore(5)

# Directory for image downloads
output_dir = "image_downloads"
os.makedirs(output_dir, exist_ok=True)

def is_command_line_mode():
    return any(vars(args).values())

def parse_arguments():
    parser = argparse.ArgumentParser(description="CivitAI Image Downloader")
    parser.add_argument("--timeout", type=int, help="Timeout value in seconds")
    parser.add_argument("--quality", type=int, choices=[1, 2], help="Image quality (1 for SD, 2 for HD)")
    parser.add_argument("--redownload", type=int, choices=[1, 2], help="Allow re-downloading of images (1 for Yes, 2 for No)")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4], help="Choose mode (1 for username, 2 for model ID, 3 for Model tag search, 4 for model version ID)")
    parser.add_argument("--tags", help="Tags for Model tag search (comma-separated)")
    parser.add_argument("--disable_prompt_check", choices=['y', 'n'], help="Disable prompt check (y/n)")
    parser.add_argument("--username", help="Username for mode 1")
    parser.add_argument("--model_id", help="Model ID for mode 2")
    parser.add_argument("--model_version_id", help="Model Version ID for mode 4")
    return parser.parse_args()
args = parse_arguments()

def create_option_folder(option_name, base_dir):
    option_dir = os.path.join(base_dir, option_name)
    os.makedirs(option_dir, exist_ok=True)
    return option_dir

allow_redownload = False

##########################################
# Updated download function with auto-detect extension
# and HD URL modification
##########################################
async def download_image(url, output_path, timeout_value, quality='SD'):
    logger_cid.info(f"Attempting to download: {url}")
    # If HD is requested, modify the URL to fetch the original (HD) version
    if quality == 'HD':
        url = re.sub(r"width=\d{3,4}", "original=true", url)
        
    async with semaphore:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=timeout_value, headers=headers)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                # Try to get extension from the Content-Type header
                content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
                mapping = {
                    "image/png": ".png",
                    "image/jpeg": ".jpeg",
                    "image/jpg": ".jpg",
                    "image/webp": ".webp",
                    "video/mp4": ".mp4",
                    "video/webm": ".webm"
                }
                file_extension = mapping.get(content_type, None)
                
                # Use the async iterator to get the first chunk from the stream
                byte_iter = response.aiter_bytes()
                try:
                    first_chunk = await anext(byte_iter)
                except StopAsyncIteration:
                    first_chunk = b""
                
                # If no valid extension from Content-Type, try to detect from the first chunk.
                if file_extension is None:
                    file_extension = detect_extension(first_chunk)
                    if file_extension is None:
                        # Fallback: default to png for HD and jpeg for SD
                        file_extension = ".png" if quality == 'HD' else ".jpeg"
                
                # If output_path already has an extension, replace it; otherwise, append the detected extension.
                if re.search(r'\.\w+$', output_path):
                    output_path_with_extension = re.sub(r'\.\w+$', file_extension, output_path)
                else:
                    output_path_with_extension = output_path + file_extension
                
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {output_path_with_extension}")
                
                with open(output_path_with_extension, "wb") as file:
                    file.write(first_chunk)
                    progress_bar.update(len(first_chunk))
                    async for chunk in byte_iter:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
                progress_bar.close()
                logger_cid.info(f"Successfully downloaded: {output_path_with_extension}")
                return True, None
        except Exception as e:
            reason = str(e)
            if isinstance(e, httpx.RequestError) or isinstance(e, httpx.ConnectError):
                reason = "Network error while downloading the image. Please check your internet connection and try again"
            elif isinstance(e, httpx.HTTPStatusError):
                reason = f"Error downloading the image. Server response: {e.response.status_code} Please try again later."
            elif isinstance(e, ConnectionResetError):
                reason = "The connection to the server was closed unexpectedly. This could be a temporary network problem. Please try again later"
            logger_cid.error(f"Error downloading {url}: {reason}")
            return False, reason

##########################################
# Async function to write meta data to a text file.
##########################################
async def write_meta_data(meta, output_path, image_id, username):
    try:
        if not meta or all(value == '' for value in meta.values()):
            output_path = output_path.replace(".txt", "_no_meta.txt")
            url = f"https://civitai.com/images/{image_id}?username={username}"
            async with aiofiles.open(output_path, "w", encoding='utf-8') as f:
                await f.write(f"No metadata available.\nURL: {url}\n")
        else:
            async with aiofiles.open(output_path, "w", encoding='utf-8') as f:
                for key, value in meta.items():
                    await f.write(f"{key}: {value}\n")
                await f.flush()  # Ensure data is written to disk
        return True
    except Exception as e:
        logger_cid.error(f"Error writing metadata: {str(e)}")
        return False

TRACKING_JSON_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "downloaded_images.json")

downloaded_images_lock = Lock()
tag_model_mapping_lock = Lock()

def load_downloaded_images():
    try:
        with open(TRACKING_JSON_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def check_if_image_downloaded(image_id, image_path, quality='SD'):
    with downloaded_images_lock:
        image_key = f"{image_id}_{quality}"
        if image_key in downloaded_images:
            existing_path = downloaded_images[image_key].get("path")
            if existing_path == image_path:
                return True
        return False

def mark_image_as_downloaded(image_id, image_path, quality='SD', tags=None, url=None):
    with downloaded_images_lock:
        image_key = f"{image_id}_{quality}"
        current_date = datetime.now().strftime("%Y-%m-%d - %H:%M")

        merged_tags = list(set(downloaded_images.get(image_key, {}).get("tags", [])) | set(tags or []))
        
        downloaded_images[image_key] = {
            "path": image_path,
            "quality": quality,
            "download_date": current_date,
            "tags": merged_tags,
            "url": url
        }
        json_data_string = json.dumps(downloaded_images[image_key], indent=4)
        with open(TRACKING_JSON_FILE, "w") as file:
            json.dump(downloaded_images, file, indent=4)
        
        logger_cid.info(f"Marked as downloaded: {image_id} at {image_path}")    

SOURCE_MISSING_MESSAGE_SHOWN = False
NEW_IMAGES_DOWNLOADED = False

# Customization of the manual_copy function
def manual_copy(src, dst):
    global SOURCE_MISSING_MESSAGE_SHOWN, NEW_IMAGES_DOWNLOADED
    # Check whether the source file exists
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst)  # Copies the file and retains the metadata
            NEW_IMAGES_DOWNLOADED = True
            return True, dst  # Returns True if the copy was successful
        except Exception as e:
            print(f"Error copying the file {src} to {dst}. Error: {e}")
            return False  # Returns False if an error has occurred
    else:
        if not SOURCE_MISSING_MESSAGE_SHOWN:
            print(f"Source file {src} does not exist. Copy skipped.")
            SOURCE_MISSING_MESSAGE_SHOWN = True
        return False  # Also returns False if the source file does not exist

def clean_and_shorten_path(path, max_total_length=260, max_component_length=80):
    # Replace %20 with a space
    path = path.replace("%20", " ")
    path = path.replace("%2B", "+")
    # Replace non-permitted characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        path = path.replace(char, '_')

    # Remove control characters
    path = re.sub(r'[\x00-\x1f\x7f]', '', path)

    # Remove trailing spaces or dots
    path = path.rstrip('. ')

    # Separate the path into directory and file name
    dir_name, file_name = os.path.split(path)

    # Shorten the file name if necessary
    if len(file_name) > max_component_length:
        file_name = file_name[:max_component_length - 3] + "_"

    # Shorten the directory name if necessary
    if len(dir_name) > max_total_length - len(file_name):
        dir_name = dir_name[:max_total_length - len(file_name) - 3] + "_"

    # Reassemble the path
    shortened_path = os.path.join(dir_name, file_name)

    # Ensure that the total path does not exceed the maximum length
    if len(shortened_path) > max_total_length:
        excess_length = len(shortened_path) - max_total_length
        shortened_path = shortened_path[:-excess_length].rstrip('. ')

    return shortened_path

def is_file_locked(filepath):
    """Check if a file is locked by another process (Windows-specific)"""
    if os.name != 'nt':
        return False
        
    try:
        fd = os.open(filepath, os.O_RDWR|os.O_EXCL)
        os.close(fd)
    except OSError:
        return True
    return False
    
def safe_move(src, dst, max_retries=15, delay=1.0):
    """Robust file moving with longer delays and more retries"""
    for attempt in range(1, max_retries+1):
        try:
            # Check if source exists before moving
            if not os.path.exists(src):
                logger_cid.warning(f"Source file {src} does not exist")
                return False
                
            shutil.move(src, dst)
            return True
        except (PermissionError, OSError) as e:
            if attempt < max_retries:
                logger_cid.warning(f"Retry {attempt}/{max_retries} for {src}")
                time.sleep(delay * attempt)  # Exponential backoff
            else:
                logger_cid.error(f"Failed to move {src} after {max_retries} attempts")
                # Attempt copy-then-delete as fallback
                try:
                    shutil.copy2(src, dst)
                    os.remove(src)
                    return True
                except Exception as copy_error:
                    logger_cid.error(f"Fallback copy failed: {copy_error}")
                raise
    return False
    
def move_to_invalid_meta(src, model_dir):
    invalid_meta_dir = os.path.join(model_dir, 'invalid_meta')
    os.makedirs(invalid_meta_dir, exist_ok=True)
    new_dst = os.path.join(invalid_meta_dir, os.path.basename(src))
    try:
        if safe_move(src, new_dst):
            logger_cid.info(f"Moved invalid meta file to {new_dst}")
    except Exception as e:
        logger_cid.error(f"Critical error moving {src}: {str(e)}")
    return new_dst

def sort_images_by_model_name(model_dir):
    global NEW_IMAGES_DOWNLOADED
    no_meta_dir = os.path.join(model_dir, 'no_meta_data')
    os.makedirs(no_meta_dir, exist_ok=True)

    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        logger_cid.info(f"No files to process in directory: {model_dir}")
        return

    all_files = os.listdir(model_dir)
    meta_files = [f for f in all_files if f.endswith(('_meta.txt', '_meta_no_meta.txt'))]

    for meta_file in meta_files:
        meta_path = os.path.join(model_dir, meta_file)
        base_name = meta_file.replace('_meta_no_meta.txt', '').replace('_meta.txt', '').strip()

        try:
            # Read metadata file content
            with open(meta_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Handle no metadata case
            if "No metadata available for this image." in content or meta_file.endswith('_meta_no_meta.txt'):
                image_moved = False
                for ext in ['.jpeg', '.png']:
                    image_path = os.path.join(model_dir, base_name + ext)
                    if os.path.exists(image_path):
                        try:
                            safe_move(image_path, os.path.join(no_meta_dir, os.path.basename(image_path)))
                            image_moved = True
                            break
                        except Exception as e:
                            logger_cid.error(f"Failed to move image {image_path}: {e}")
                            continue

                if not image_moved:
                    logger_cid.info(f"No image found for metadata file {meta_file} (This is expected for images that didn't pass the prompt check)")

                # Move metadata file to no_meta_dir
                try:
                    safe_move(meta_path, os.path.join(no_meta_dir, meta_file))
                except Exception as e:
                    logger_cid.error(f"Failed to move metadata file {meta_file}: {e}")

            # Handle valid metadata case
            else:
                model_name_found = False
                model_name = None
                for line in content.split('\n'):
                    if "Model:" in line:
                        model_name = line.split(":")[1].strip()
                        model_name_found = True
                        break

                if model_name_found:
                    model_name = clean_and_shorten_path(model_name)
                    target_dir = os.path.join(model_dir, model_name)
                    os.makedirs(target_dir, exist_ok=True)
                    process_image_and_meta(model_dir, meta_file, target_dir, valid_meta=True)
                else:
                    process_image_and_meta(model_dir, meta_file, model_dir, valid_meta=False)

        except Exception as e:
            logger_cid.error(f"Error processing metadata file {meta_file}: {e}")
            continue

    # Clean up orphaned images
    downloaded_images_in_dir = [f for f in all_files if f.endswith(('.jpeg', '.png')) and os.path.join(model_dir, f) in download_stats["downloaded"]]
    for file in downloaded_images_in_dir:
        base_name = file.rsplit('.', 1)[0]
        if not any(meta for meta in meta_files if meta.startswith(base_name)):
            logger_cid.warning(f"Orphaned image found: {os.path.join(model_dir, file)}")
            try:
                os.remove(os.path.join(model_dir, file))
            except Exception as e:
                logger_cid.error(f"Failed to remove orphaned image {file}: {e}")

    if meta_files:
        NEW_IMAGES_DOWNLOADED = True

def process_image_and_meta(model_dir, meta_file, target_dir, valid_meta):
    base_name = meta_file.replace('_meta.txt', '').replace('_meta_no_meta.txt', '')
    meta_path = os.path.join(model_dir, meta_file)
    
    # First process metadata file
    new_meta_path = None
    try:
        if valid_meta:
            new_meta_path = os.path.join(target_dir, meta_file)
            if safe_move(meta_path, new_meta_path):
                logger_cid.info(f"Moved metadata {meta_file}")
        else:
            new_meta_path = move_to_invalid_meta(meta_path, model_dir)
    except Exception as e:
        logger_cid.error(f"Metadata move failed: {e}")
        return None, None

    # Then process image file
    new_image_path = None
    extensions = ['.jpeg', '.png']
    for ext in extensions:
        image_path = os.path.join(model_dir, base_name + ext)
        if os.path.exists(image_path):
            try:
                if valid_meta:
                    new_image_path = os.path.join(target_dir, os.path.basename(image_path))
                    if safe_move(image_path, new_image_path):
                        logger_cid.info(f"Moved image {base_name}{ext}")
                else:
                    new_image_path = move_to_invalid_meta(image_path, model_dir)
                break
            except Exception as e:
                logger_cid.error(f"Image move failed: {e}")
                continue

    return new_image_path, new_meta_path

visited_pages = set()

async def search_models_by_tag(tag, failed_search_requests=[]):
    base_url = f"https://civitai.com/api/v1/models?tag={tag}&nsfw=true"
    model_id = set()
    async with httpx.AsyncClient() as client:
        async with semaphore:
            while base_url:
                try:
                    response = await client.get(base_url, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get('items', [])                        
                        if items:  # If items list is not empty
                            for model in items:
                                model_id.add(model['id'])
                        else:  # If items list is empty
                            print(f"No models found for the tag '{tag}'.")
                            return model_id  # Return the empty set
                        metadata = data.get('metadata', {})
                        nextPage = metadata.get('nextPage', None)
                        base_url = nextPage if nextPage else None
                    else:
                        logger_cid.error(f"Server response: {response.status_code} for URL: {base_url}")
                        print(f"Unexpected server response: {response.status_code}. Please try again later.")
                        failed_search_requests.append(base_url)
                        break
                except httpx.RequestError as e:
                    logger_cid.exception(f"Request error occurred while fetching models for tag '{tag}': {e}")
                    print("A connection error occurred. Please check your internet connection and try again.")
                    failed_searchRequests.append(base_url)
                    break
        return model_id

tag_model_mapping = {}

async def download_images_for_model_with_tag_check(model_ids, option_folder, timeout_value, quality='SD', tag_to_check=None, tag_dir_name=None, sanitized_tag_dir_name=None, disable_prompt_check=False, allow_redownload=2):
    global NEW_IMAGES_DOWNLOADED, download_stats
    failed_urls = []
    images_without_meta = 0
    tasks = []
    total_api_items = 0
    total_downloaded_items = 0

    if tag_to_check is None:
        tag_to_check = tag_dir_name

    for model_id in model_ids:
        url = f"{base_url}?modelId={str(model_id)}&nsfw=X"
        visited_pages = set()

        while url:
            if url in visited_pages:
                logger_cid.info(f"URL {url} already visited. Ending loop.")
                break
            visited_pages.add(url)
            async with httpx.AsyncClient() as client:
                async with semaphore:
                    try:
                        response = await client.get(url, timeout=timeout_value, headers=headers)
                        if response.status_code == 200:
                            data = response.json()
                            items = data.get('items', [])
                            total_api_items += len(items)

                            # Create directories with exist_ok=True
                            tag_dir = os.path.join(option_folder, sanitized_tag_dir_name)
                            os.makedirs(tag_dir, exist_ok=True)
                            
                            model_dir = os.path.join(tag_dir, f"model_{model_id}")
                            os.makedirs(model_dir, exist_ok=True)

                            with tag_model_mapping_lock:
                                if tag_dir_name not in tag_model_mapping:
                                    tag_model_mapping[tag_dir_name] = []
                                model_name = items[0]["meta"].get("Model", "unknown_model") if items and isinstance(items[0].get("meta"), dict) else "unknown_model"
                                tag_model_mapping[tag_dir_name].append((model_id, model_name))

                            current_batch_tasks = []
                            for item in items:
                                item_meta = item.get("meta") or {}
                                prompt = item_meta.get("prompt", "").lower()
                                tag_check_passed = (
                                    disable_prompt_check or 
                                    not tag_to_check or 
                                    all(word in prompt for word in tag_to_check.lower().split("_"))
                                )

                                if tag_check_passed:
                                    image_id = item['id']
                                    image_url = item['url']
                                    ext = ".png" if quality == 'HD' else ".jpeg"
                                    image_path = os.path.join(model_dir, f"{image_id}{ext}")

                                    if allow_redownload == 2 and check_if_image_downloaded(str(image_id), image_path, quality):
                                        continue

                                    current_batch_tasks.append(
                                        download_image(image_url, image_path, timeout_value, quality)
                                    )

                            # Process batch downloads with progress tracking
                            batch_results = []
                            if current_batch_tasks:
                                batch_results = await asyncio.gather(*current_batch_tasks)
                                await asyncio.sleep(1)  # Add brief pause between batches

                            # Process results with error handling
                            for idx, (success, reason) in enumerate(batch_results):
                                item = items[idx]
                                image_id = item['id']
                                image_path = os.path.join(model_dir, f"{image_id}{'.png' if quality == 'HD' else '.jpeg'}")

                                if success:
                                    NEW_IMAGES_DOWNLOADED = True
                                    total_downloaded_items += 1
                                    tags = [tag_to_check] if tag_to_check else []
                                    mark_image_as_downloaded(str(image_id), image_path, quality, tags=tags, url=item['url'])
                                    download_stats["downloaded"].append(image_path)
                                    
                                    # Write metadata with async file handling
                                    meta_path = os.path.join(model_dir, f"{image_id}_meta.txt")
                                    await write_meta_data(item.get("meta"), meta_path, image_id, item.get('username', 'unknown'))
                                    
                                    if not item.get("meta"):
                                        images_without_meta += 1
                                else:
                                    logger_cid.error(f"Failed to download image {image_id}: {reason}")
                                    download_stats["skipped"].append((item['url'], reason))

                            # Add cleanup before sorting
                            if sys.platform == 'win32':
                                await asyncio.sleep(2)  # Extra time for Windows file handles
                            
                            try:
                                sort_images_by_model_name(model_dir)
                            except Exception as sort_error:
                                logger_cid.error(f"Error sorting files in {model_dir}: {str(sort_error)}")

                            # Pagination handling
                            if data.get('metadata', {}).get('nextPage'):
                                url = data['metadata']['nextPage']
                                await asyncio.sleep(3)
                            else:
                                url = None

                    except Exception as e:
                        logger_cid.error(f"Error processing URL {url}: {str(e)}")
                        failed_urls.append(url)
                        continue

    return tasks, failed_urls, images_without_meta, sanitized_tag_dir_name, total_api_items, total_downloaded_items

def sort_images_by_tag(option_folder, tag_model_mapping):
    with tag_model_mapping_lock:
        for tag, model_ids in tag_model_mapping.items():
            sanitized_tag = tag.replace(" ", "_")
            tag_dir = os.path.join(option_folder, sanitized_tag)
            if not os.listdir(tag_dir):
                print(f"No images found for the tag: {tag}")

def write_summary_to_csv(tag, downloaded_images, option_folder, tag_model_mapping):
    with tag_model_mapping_lock:
        tag_dir = os.path.join(option_folder, tag.replace(" ", "_"))
        for model_info in tag_model_mapping.get(tag, []):
            model_id, model_name = model_info
            model_dir = os.path.join(tag_dir, f"model_{(model_id)}")
        
            # Check if the model directory exists
            if not os.path.exists(model_dir):
                print(f"The {model_dir} directory does not exist. Skip the creation of the CSV file.")
                continue
            csv_file = os.path.join(model_dir, f"{tag.replace(' ', '_')}_summary_{datetime.now().strftime('%Y%m%d')}.csv")
            with open(csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Current Tag", "Previously Downloaded Tag", "Image Path", "Download URL"])
                for image_id, image_info in downloaded_images.items():
                    # Here it is assumed that the "tags" are stored in the "downloaded_images" dictionary
                    if tag in image_info.get("tags", []):
                        for prev_tag in image_info.get("tags", []):
                            if prev_tag != tag:
                                relative_path = os.path.relpath(image_info["path"], model_dir)
                                writer.writerow([tag, prev_tag, relative_path, image_info["url"]])

failed_identifiers = []  # List for saving failed usernames and model IDs

async def is_valid_username(username):
    url = f"{base_url}?username={username.strip()}&nsfw=X"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code == 500:
                response_data = response.json()
                if 'error' in response_data and response_data['error'] == "User not found":
                    return False, "Username not found"
            return True, None
        except httpx.RequestError as e:
            return False, f"Network error: {e}"
        except json.JSONDecodeError as e:
            return False, f"Error decoding response: {e}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

async def is_valid_model_id(identifier):
    url = f"{base_url}?modelId={str(identifier)}&nsfw=X"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code == 500:
                # If the modelId fails, try modelVersionId
                url = f"{base_url}?modelVersionId={str(identifier)}&nsfw=X"
                response = await client.get(url, headers=headers)
                if response.status_code == 500:
                    return False, f"Invalid input syntax for model ID or model version ID: {identifier}"
            elif response.status_code == 304:
                response_data = response.json()
                if not response_data['items']:
                    return False, f"No items found for model ID or model version ID: {identifier}"
            return True, None
        except httpx.RequestError as e:
            return False, f"Network error: {e}"
        except json.JSONDecodeError as e:
            return False, f"Error decoding response: {e}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

async def is_valid_model_version_id(model_version_id):
    url = f"{base_url}?modelVersionId={str(model_version_id)}&nsfw=X"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code == 500:
                return False, f"Invalid input syntax for model version ID: {model_version_id}"
            elif response.status_code == 304:
                response_data = response.json()
                if not response_data['items']:
                    return False, f"No items found for model version ID: {model_version_id}"
            return True, None
        except httpx.RequestError as e:
            return False, f"Network error: {e}"
        except json.JSONDecodeError as e:
            return False, f"Error decoding response: {e}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

def get_url_for_identifier(identifier, identifier_type):
    base_url = "https://civitai.com/api/v1/images"
    if identifier_type == 'model':
        return f"{base_url}?modelId={str(identifier)}&nsfw=X"
    elif identifier_type == 'modelVersion':
        return f"{base_url}?modelVersionId={str(identifier)}&nsfw=X"
    elif identifier_type == 'username':
        return f"{base_url}?username={identifier.strip()}&nsfw=X&sort=Newest"
    else:
        raise ValueError("Invalid identifier_type. Should be 'model', 'modelVersion', or 'username'.")

async def download_images(identifier, option_folder, identifier_type, timeout_value, quality='SD', allow_redownload=2):
    global NEW_IMAGES_DOWNLOADED, download_stats
    valid, error_message = True, None
    if identifier_type == 'username':
        valid, error_message = await is_valid_username(identifier)
    elif identifier_type == 'model':
        valid, error_message = await is_valid_model_id(identifier)
    elif identifier_type == 'modelVersion':
        valid, error_message = await is_valid_model_version_id(identifier)
    if not valid:
        logger_cid.warning(f"Skipping: {error_message}")
        failed_identifiers.append((identifier_type, identifier))
        return [], 0, 0, 0

    url = get_url_for_identifier(identifier, identifier_type)
    failed_urls = []
    images_without_meta = 0
    total_items = 0
    total_downloaded = 0

    # Define dir_name here, outside the while loop
    if identifier_type in ['model', 'modelVersion']:
        dir_name = os.path.join(option_folder, f"{identifier_type}_{identifier}")
    elif identifier_type == 'username':
        dir_name = os.path.join(option_folder, identifier.strip())
    else:
        logger_cid.error(f"Invalid identifier_type: {identifier_type}")
        return [], 0, 0, 0

    os.makedirs(dir_name, exist_ok=True)

    while url:
        if url in visited_pages:
            logger_cid.info(f"URL {url} already visited. Ending loop.")
            break
        visited_pages.add(url)     
        async with httpx.AsyncClient() as client:
            async with semaphore:
                try:
                    response = await client.get(url, timeout=timeout_value)
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get('items', [])
                        logger_cid.info(f"Received {len(items)} items from API for {identifier_type} {identifier}")
                        total_items += len(items)

                        tasks = []
                        for item in items:
                            image_id = item['id']
                            image_url = item['url']
                            image_extension = ".png" if quality == 'HD' else ".jpeg"
                            image_path = os.path.join(dir_name, f"{image_id}{image_extension}")

                            if allow_redownload == 2 and check_if_image_downloaded(str(image_id), image_path, quality):
                                continue

                            task = download_image(image_url, image_path, timeout_value, quality)
                            tasks.append(task)

                        download_results = await asyncio.gather(*tasks)
                        total_downloaded += sum(1 for result, _ in download_results if result)

                        for idx, (download_success, reason) in enumerate(download_results):
                            image_id = items[idx]['id']
                            if download_success:
                                NEW_IMAGES_DOWNLOADED = True
                                image_path = os.path.join(dir_name, f"{image_id}{'.png' if quality == 'HD' else '.jpeg'}")
                                mark_image_as_downloaded(str(image_id), image_path, quality)
                                download_stats["downloaded"].append(image_path)
                            else:
                                download_stats["skipped"].append((items[idx]['url'], reason))
                            
                            meta_output_path = os.path.join(dir_name, f"{image_id}_meta.txt")
                            await write_meta_data(items[idx].get("meta"), meta_output_path, image_id, items[idx].get('username', 'unknown'))
                            if not items[idx].get("meta"):
                                images_without_meta += 1
                        
                        metadata = data['metadata']
                        next_page = metadata.get('nextPage')

                        if next_page:
                            url = next_page
                            await asyncio.sleep(3)
                        else:
                            break
                except Exception as e:
                    logger_cid.error(f"Error processing URL {url}: {str(e)}")
                    failed_urls.append(url)
                    continue

    if identifier_type in ['model', 'modelVersion', 'username']:
        sort_images_by_model_name(dir_name)

    logger_cid.info(f"Total items from API: {total_items}, Total downloaded: {total_downloaded}")
    return failed_urls, images_without_meta, total_items, total_downloaded

def print_download_statistics():
    print("\n")
    print(f"Number of downloaded images: {len(download_stats['downloaded'])}")
    print(f"Number of skipped images: {len(download_stats['skipped'])}")

    if download_stats['skipped']:
        print("\nReasons for skipping images:")
        reasons = {}
        for _, reason in download_stats['skipped']:
            if reason in reasons:
                reasons[reason] += 1
            else:
                reasons[reason] = 1
        
        for reason, count in reasons.items():
            print(f"- {reason}: {count} times")

# Main async function
async def main():
    global downloaded_images, download_stats
    downloaded_images = load_downloaded_images()
    download_stats = {
        "downloaded": [],
        "skipped": [],
    }

    # Check if we're in mixed mode
    provided_args = [arg for arg in vars(args) if getattr(args, arg) is not None]
    if provided_args and len(provided_args) < len(vars(args)):
        print("Mixed mode detected. Some arguments provided via command-line, others will be prompted.")
        logger_cid.info("Running in mixed mode. Some arguments provided via command-line, others will be prompted.")

    # Check for mismatched arguments
    check_mismatched_arguments()

    timeout_value = get_timeout_value()
    quality = get_quality()
    allow_redownload = get_redownload_option()

    failed_search_requests = []
    failed_urls = []
    images_without_meta = 0
    total_api_items = 0
    total_downloaded_items = 0

    choice = get_mode_choice()
    tasks = []

    if choice == "1":
        usernames = get_usernames()
        option_folder = create_option_folder('Username_Search', output_dir)
        tasks.extend([download_images(username.strip(), option_folder, 'username', timeout_value, quality, allow_redownload) for username in usernames])

    elif choice == "2":
        model_ids = get_model_ids()
        option_folder = create_option_folder('Model_ID_Search', output_dir)
        tasks.extend([download_images(model_id.strip(), option_folder, 'model', timeout_value, quality, allow_redownload) for model_id in model_ids])

    elif choice == "3":
        tags = get_tags()
        disable_prompt_check = get_disable_prompt_check()
        option_folder = create_option_folder('Model_Tag_Search', output_dir)

        for tag in tags:
            sanitized_tag_dir_name = tag.replace(" ", "_")
            model_ids = await search_models_by_tag(tag.replace("_", "%20"), failed_search_requests)
            tag_to_check = None if disable_prompt_check else tag
            tasks_for_tag, failed_urls_for_tag, images_without_meta_for_tag, sanitized_tag_dir_name_for_tag, api_items, downloaded_items = await download_images_for_model_with_tag_check(model_ids, option_folder, timeout_value, quality, tag_to_check, tag, sanitized_tag_dir_name, disable_prompt_check, allow_redownload)
            tasks.extend(tasks_for_tag)
            failed_urls.extend(failed_urls_for_tag)
            images_without_meta += images_without_meta_for_tag
            total_api_items += api_items
            total_downloaded_items += downloaded_items

        # Sort images into tag-related folders
        sort_images_by_tag(option_folder, tag_model_mapping)

        for tag, model_ids in tag_model_mapping.items():
            write_summary_to_csv(tag, downloaded_images, option_folder, tag_model_mapping)

    elif choice == "4":
        model_version_ids = get_model_version_ids()
        option_folder = create_option_folder('Model_Version_ID_Search', output_dir)
        tasks.extend([download_images(model_version_id.strip(), option_folder, 'modelVersion', timeout_value, quality, allow_redownload) for model_version_id in model_version_ids])

    else:
        logger_cid.error("Invalid choice!")
        return

    # Execute all collected tasks
    results = await asyncio.gather(*tasks)

    # Extract failed_urls, images_without_meta, and download statistics from the results
    for result in results:
        failed_urls.extend(result[0])
        images_without_meta += result[1]
        total_api_items += result[2]
        total_downloaded_items += result[3]

    for tag, model_ids in tag_model_mapping.items():
        write_summary_to_csv(tag, downloaded_images, option_folder, tag_model_mapping)

    if failed_urls:
        logger_cid.info("Retrying failed URLs...")
        for url in failed_urls:
            await download_image(url, option_folder, timeout_value=timeout_value, quality=quality)

    # Attempt to retry failed search requests
    if failed_search_requests:
        logger_cid.info("Retrying failed search requests...")
        for url in failed_search_requests:
            tag = url.split("tag=")[-1]
            await search_models_by_tag(tag, [])

    if images_without_meta > 0:
        logger_cid.info(f"{images_without_meta} images have no meta data.")

    logger_cid.info(f"Total API items: {total_api_items}, Total downloaded: {total_downloaded_items}")
    print(f"Total API items: {total_api_items}, Total downloaded: {total_downloaded_items}")
    print_download_statistics()

# Helper functions for main
def check_mismatched_arguments():
    if args.mode:
        if args.mode == 1 and (args.model_id or args.model_version_id or args.tags):
            print("Warning: --model_id, --model_version_id, and --tags are not used in username mode. These arguments will be ignored.")
            logger_cid.warning("Warning: --model_id, --model_version_id, and --tags are not used in username mode. These arguments will be ignored.")
        elif args.mode == 2 and (args.username or args.model_version_id or args.tags):
            print("Warning: --username, --model_version_id, and --tags are not used in model ID mode. These arguments will be ignored.")
            logger_cid.warning("Warning: --username, --model_version_id, and --tags are not used in model ID mode. These arguments will be ignored.")
        elif args.mode == 3 and (args.username or args.model_id or args.model_version_id):
            print("Warning: --username, --model_id, and --model_version_id are not used in tag search mode. These arguments will be ignored.")
            logger_cid.warning("Warning: --username, --model_id, and --model_version_id are not used in tag search mode. These arguments will be ignored.")
        elif args.mode == 4 and (args.username or args.model_id or args.tags):
            print("Warning: --username, --model_id, and --tags are not used in model version ID mode. These arguments will be ignored.")
            logger_cid.warning("Warning: --username, --model_id, and --tags are not used in model version ID mode. These arguments will be ignored.")

def get_timeout_value():
    if args.timeout:
        return args.timeout
    else:
        timeout_input = input("Enter timeout value (in seconds): ")
        if timeout_input.isdigit() and int(timeout_input) > 0:
            return int(timeout_input)
        else:
            logger_cid.warning("Invalid timeout value. Using default value of 60 seconds.")
            return 60

def get_quality():
    if args.quality:
        return 'HD' if args.quality == 2 else 'SD'
    else:
        quality_choice = input("Choose image quality (1 for SD, 2 for HD): ")
        if quality_choice == '2':
            return 'HD'
        elif quality_choice == '1':
            return 'SD'
        else:
            logger_cid.warning("Invalid quality choice. Using default quality SD.")
            return 'SD'

def get_redownload_option():
    if args.redownload:
        return args.redownload
    else:
        allow_redownload_choice = input("Allow re-downloading of images already tracked (1 for Yes, 2 for No) [default: 2]: ")
        if allow_redownload_choice == '1':
            return 1
        elif allow_redownload_choice == '2' or allow_redownload_choice.strip() == '':
            return 2
        else:
            logger_cid.warning("Invalid choice. Using default value (2 - No).")
            return 2

def get_mode_choice():
    if args.mode:
        return str(args.mode)
    else:
        return input("Choose mode (1 for username, 2 for model ID, 3 for Model tag search, 4 for model version ID): ")

def get_usernames():
    if args.username:
        return [args.username]
    else:
        return input("Enter username: ").split(",")

def get_model_ids():
    if args.model_id:
        return [args.model_id]
    else:
        while True:
            model_ids_input = input("Enter model ID: ")
            model_ids = model_ids_input.split(",")
            if all(model_id.strip().isdigit() for model_id in model_ids):
                return model_ids
            else:
                logger_cid.warning("Invalid input. Please enter only numeric model IDs.")

def get_tags():
    if args.tags:
        return [tag.strip().replace(" ", "_") for tag in args.tags.split(',')]
    else:
        tags_input = input("Enter tags (comma-separated): ")
        return [tag.strip().replace(" ", "_") for tag in tags_input.split(',')]

def get_disable_prompt_check():
    if args.disable_prompt_check is not None:
        return args.disable_prompt_check.lower() == 'y'
    else:
        return input("Disable prompt check? (y/n): ").lower() in ['y', 'yes']

def get_model_version_ids():
    if args.model_version_id:
        return [args.model_version_id]
    else:
        while True:
            model_version_ids_input = input("Enter model version ID: ")
            model_version_ids = model_version_ids_input.split(",")
            if all(model_version_id.strip().isdigit() for model_version_id in model_version_ids):
                return model_version_ids
            else:
                logger_cid.warning("Invalid input. Please enter only numeric model version IDs.")

if __name__ == "__main__":
    args = parse_arguments()
    
    if is_command_line_mode():
        print("Running in command-line mode.")
        logger_cid.info("Running in command-line mode")
    else:
        print("Running in interactive mode.")
        logger_cid.info("Running in interactive mode")
    
    asyncio.run(main())

    if failed_identifiers:
        logger_cid.warning("Failed identifiers:")
        for id_type, id_value in failed_identifiers:
            logger_cid.warning(f"{id_type}: {id_value}")

    logger_cid.info("Image download completed.")
