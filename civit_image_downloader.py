import httpx
import os
import asyncio
import json
from tqdm import tqdm
import shutil
import re
from datetime import datetime
import logging
import csv
from threading import Lock
import argparse
from configparser import ConfigParser
from pathlib import Path

# --- Configuration ---
CONFIG_FILE = Path(__file__).resolve().parent / "config.ini"

def load_config():
    config = ConfigParser()
    if not CONFIG_FILE.exists():
        # Create a default configuration file
        config['DEFAULT'] = {
            'output_dir': 'image_downloads',
            'default_timeout': '60',
            'default_quality': 'SD',  # SD or HD
            'max_concurrent_downloads': '5',
            'retry_attempts': '3',
            'retry_delay': '5', #seconds
            'log_level': 'INFO',
        }
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
    config.read(CONFIG_FILE)
    return config

config = load_config()

# --- Constants ---
BASE_URL = "https://civitai.com/api/v1/images"
TIMEOUT = int(config['DEFAULT']['default_timeout'])
DEFAULT_QUALITY = config['DEFAULT']['default_quality']
MAX_CONCURRENT_DOWNLOADS = int(config['DEFAULT']['max_concurrent_downloads'])
RETRY_ATTEMPTS = int(config['DEFAULT']['retry_attempts'])
RETRY_DELAY = int(config['DEFAULT']['retry_delay'])
OUTPUT_DIR = Path(config['DEFAULT']['output_dir'])
TRACKING_JSON_FILE = Path(__file__).resolve().parent / "downloaded_images.json"

# --- Setup logging ---
logger_cid = logging.getLogger('cid')
logger_cid.setLevel(getattr(logging, config['DEFAULT']['log_level'].upper()))

log_file_path = Path(__file__).resolve().parent / "civit_image_downloader_log.txt"
file_handler_cid = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler_cid.setLevel(logging.DEBUG)  # Log everything to file
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler_cid.setFormatter(formatter)
logger_cid.addHandler(file_handler_cid)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # INFO level and above for console
console_handler.setFormatter(formatter)
logger_cid.addHandler(console_handler)

# --- Locks ---
downloaded_images_lock = Lock()
tag_model_mapping_lock = Lock()

# --- Global Variables ---
downloaded_images = {}  # Initialize here
download_stats = {
    "downloaded": [],
    "skipped": [],
}
semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
visited_pages = set()
tag_model_mapping = {}
failed_identifiers = []  # For saving failed usernames and model IDs



# --- Helper Functions ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="CivitAI Image Downloader")
    parser.add_argument("--timeout", type=int, help="Timeout value in seconds", default=TIMEOUT)
    parser.add_argument("--quality", type=int, choices=[1, 2], help="Image quality (1 for SD, 2 for HD)", default=1 if DEFAULT_QUALITY == 'SD' else 2)
    parser.add_argument("--redownload", type=int, choices=[1, 2], help="Allow re-downloading of images (1 for Yes, 2 for No)", default=2)
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4], help="Choose mode (1 for username, 2 for model ID, 3 for Model tag search, 4 for model version ID)")
    parser.add_argument("--tags", help="Tags for Model tag search (comma-separated)")
    parser.add_argument("--disable_prompt_check", choices=['y', 'n'], help="Disable prompt check (y/n)")
    parser.add_argument("--username", help="Username for mode 1")
    parser.add_argument("--model_id", help="Model ID for mode 2")
    parser.add_argument("--model_version_id", help="Model Version ID for mode 4")
    return parser.parse_args()


def is_command_line_mode():
    return any(vars(args).values())

def create_option_folder(option_name, base_dir):
    option_dir = base_dir / option_name
    option_dir.mkdir(parents=True, exist_ok=True)
    return option_dir


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
            return existing_path == str(image_path)
        return False

def mark_image_as_downloaded(image_id, image_path, quality='SD', tags=None, url=None):
    with downloaded_images_lock:
        image_key = f"{image_id}_{quality}"
        current_date = datetime.now().strftime("%Y-%m-%d - %H:%M")
        merged_tags = list(set(downloaded_images.get(image_key, {}).get("tags", [])) | set(tags or []))

        downloaded_images[image_key] = {
            "path": str(image_path),
            "quality": quality,
            "download_date": current_date,
            "tags": merged_tags,
            "url": url
        }
        with open(TRACKING_JSON_FILE, "w") as file:
            json.dump(downloaded_images, file, indent=4)
        logger_cid.info(f"Marked as downloaded: {image_id} at {image_path}")


def clean_and_shorten_path(path, max_total_length=260, max_component_length=80):
    path = path.replace("%20", " ").replace("%2B", "+")
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        path = path.replace(char, '_')
    path = re.sub(r'[\x00-\x1f\x7f]', '', path)
    path = path.rstrip('. ')
    dir_name, file_name = os.path.split(path)

    if len(file_name) > max_component_length:
        file_name = file_name[:max_component_length - 3] + "_"
    if len(dir_name) > max_total_length - len(file_name):
        dir_name = dir_name[:max_total_length - len(file_name) - 3] + "_"
    shortened_path = os.path.join(dir_name, file_name)

    if len(shortened_path) > max_total_length:
        excess_length = len(shortened_path) - max_total_length
        shortened_path = shortened_path[:-excess_length].rstrip('. ')
    return shortened_path

def move_to_invalid_meta(src, model_dir):
    invalid_meta_dir = model_dir / 'invalid_meta'
    invalid_meta_dir.mkdir(parents=True, exist_ok=True)
    new_dst = invalid_meta_dir / os.path.basename(src)
    try:
        shutil.move(src, new_dst)
    except Exception as e:
        print(f"Error moving the file {src} to the 'invalid_meta' folder. Error: {e}")
    return new_dst


def process_image_and_meta(model_dir, meta_file, target_dir, valid_meta):
    base_name = meta_file.replace('_meta.txt', '').replace('_meta_no_meta.txt', '')
    image_moved = False
    new_image_path = None
    extensions = ['.jpeg', '.png']

    for extension in extensions:
        image_path = model_dir / f"{base_name}{extension}"
        if image_path.exists():
            try:
                if valid_meta:
                    new_image_path = target_dir / image_path.name
                    shutil.move(image_path, new_image_path)
                else:
                    new_image_path = move_to_invalid_meta(image_path, model_dir)
                image_moved = True
                logger_cid.info(f"Moved image {image_path} to {new_image_path}")
            except Exception as e:
                logger_cid.error(f"Error moving image {image_path}: {e}")
            break

    if not image_moved:
        logger_cid.info(f"No image found for metadata file {meta_file} (This is expected for images that didn't pass the prompt check)")
        return None, None

    meta_path = model_dir / meta_file
    if valid_meta:
        new_meta_path = target_dir / meta_file
        shutil.move(meta_path, new_meta_path)
    else:
        new_meta_path = move_to_invalid_meta(meta_path, model_dir)

    return new_image_path, new_meta_path


def sort_images_by_model_name(model_dir):
    global NEW_IMAGES_DOWNLOADED
    no_meta_dir = model_dir / 'no_meta_data'
    no_meta_dir.mkdir(parents=True, exist_ok=True)

    if model_dir.exists() and any(model_dir.iterdir()):
        all_files = list(model_dir.iterdir())
        meta_files = [f.name for f in all_files if f.name.endswith('_meta.txt') or f.name.endswith('_meta_no_meta.txt')]

        for meta_file in meta_files:
            with open(model_dir / meta_file, 'r', encoding='utf-8') as file:
                content = file.read()
                base_name = meta_file.replace('_meta_no_meta.txt', '').replace('_meta.txt', '').strip()

                if "No metadata available for this image." in content or meta_file.endswith('_meta_no_meta.txt'):
                    image_moved = False
                    for ext in ['.jpeg', '.png']:
                        image_path = model_dir / f"{base_name}{ext}"
                        if image_path.exists():
                            shutil.move(image_path, no_meta_dir / image_path.name)
                            image_moved = True
                            break
                    if not image_moved:
                        logger_cid.info(f"No image found for metadata file {meta_file} (This is expected for images that didn't pass the prompt check)")
                    shutil.move(model_dir / meta_file, no_meta_dir / meta_file)

                else:
                    model_name_found = False
                    for line in content.split('\n'):
                        if "Model:" in line:
                            model_name = line.split(":")[1].strip()
                            model_name_found = True
                            break

                    if model_name_found:
                        model_name = clean_and_shorten_path(model_name)
                        target_dir = model_dir / model_name
                        target_dir.mkdir(parents=True, exist_ok=True)
                        process_image_and_meta(model_dir, meta_file, target_dir, valid_meta=True)
                    else:
                        process_image_and_meta(model_dir, meta_file, model_dir, valid_meta=False)

        # Only check for orphaned images among those that were actually downloaded.
        downloaded_images_in_dir = [f.name for f in all_files if f.name.endswith(('.jpeg', '.png')) and str(f) in download_stats["downloaded"]]
        for file in downloaded_images_in_dir:
            base_name = file.rsplit('.', 1)[0]
            if not any(meta for meta in meta_files if meta.startswith(base_name)):
                logger_cid.warning(f"Orphaned image found: {model_dir / file}")
                (model_dir / file).unlink()

        if meta_files:
          NEW_IMAGES_DOWNLOADED = True


def get_url_for_identifier(identifier, identifier_type):
    if identifier_type == 'model':
        return f"{BASE_URL}?modelId={str(identifier)}&nsfw=X"
    elif identifier_type == 'modelVersion':
        return f"{BASE_URL}?modelVersionId={str(identifier)}&nsfw=X"
    elif identifier_type == 'username':
        return f"{BASE_URL}?username={identifier.strip()}&nsfw=X&sort=Newest"
    else:
        raise ValueError("Invalid identifier_type. Should be 'model', 'modelVersion', or 'username'.")

async def is_valid_username(username):
    url = f"{BASE_URL}?username={username.strip()}&nsfw=X"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"}) # Headers inside
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
    url = f"{BASE_URL}?modelId={str(identifier)}&nsfw=X"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code == 500:
                # If the modelId fails, try modelVersionId
                url = f"{BASE_URL}?modelVersionId={str(identifier)}&nsfw=X"
                response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
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
    url = f"{BASE_URL}?modelVersionId={str(model_version_id)}&nsfw=X"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
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

# --- Core Download Functions ---

async def download_image(url, output_path, timeout_value, quality='SD'):
    logger_cid.info(f"Attempting to download: {url}")
    file_extension = ".png" if quality == 'HD' else ".jpeg"
    output_path_with_extension = output_path.with_suffix(file_extension)

    if quality == 'HD':
        url = re.sub(r"width=\d{3,4}", "original=true", url)

    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=timeout_value, headers={"User-Agent": "Mozilla/5.0"})
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {output_path_with_extension.name}")

                    with open(output_path_with_extension, "wb") as file:
                        async for chunk in response.aiter_bytes():
                            file.write(chunk)
                            progress_bar.update(len(chunk))
                    progress_bar.close()
                    logger_cid.info(f"Successfully downloaded: {output_path_with_extension}")
                    return True, None

            except httpx.TimeoutException:
                reason = "Timeout error.  The server did not respond within the specified time."
                logger_cid.error(f"Timeout error downloading {url} (attempt {attempt + 1}/{RETRY_ATTEMPTS})")
            except httpx.RequestError as e:
                reason = "Network error. Check internet connection."
                logger_cid.error(f"Request error downloading {url} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            except httpx.HTTPStatusError as e:
                 reason = f"HTTP error. Server Responded with: {e.response.status_code}"
                 logger_cid.error(f"HTTP error downloading {url} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            except ConnectionResetError:
                reason = "Connection reset. The server closed the connection unexpectedly."
                logger_cid.error(f"Connection reset error downloading {url} (attempt {attempt + 1}/{RETRY_ATTEMPTS})")
            except Exception as e:
                reason = str(e)
                logger_cid.error(f"Error downloading {url} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")

            if attempt < RETRY_ATTEMPTS - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
            else:
                return False, reason


async def write_meta_data(meta, output_path, image_id, username):
    if not meta or all(value == '' for value in meta.values()):
        output_path = output_path.with_name(output_path.stem + "_no_meta.txt")
        url = f"https://civitai.com/images/{image_id}?period=AllTime&periodMode=published&sort=Newest&view=feed&username={username}&withTags=false"
        with open(output_path, "w", encoding='utf-8') as file:
            file.write(f"No metadata available for this image.\nURL: {url}\n")
    else:
        with open(output_path, "w", encoding='utf-8') as file:
            for key, value in meta.items():
                file.write(f"{key}: {value}\n")


async def search_models_by_tag(tag, failed_search_requests=[]):
    base_url = f"https://civitai.com/api/v1/models?tag={tag}&nsfw=true"
    model_ids = set()
    async with httpx.AsyncClient() as client:
        async with semaphore:
            while base_url:
                try:
                    response = await client.get(base_url, headers={"User-Agent": "Mozilla/5.0"})
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get('items', [])
                        if items:
                            for model in items:
                                model_ids.add(model['id'])
                        else:
                            print(f"No models found for the tag '{tag}'.")
                            return model_ids  # Return empty set if no models
                        metadata = data.get('metadata', {})
                        next_page = metadata.get('nextPage')
                        base_url = next_page if next_page else None
                    else:
                        logger_cid.error(f"Server response: {response.status_code} for URL: {base_url}")
                        print(f"Unexpected server response: {response.status_code}. Please try again later.")
                        failed_search_requests.append(base_url)
                        break  # Exit loop on error
                except httpx.RequestError as e:
                    logger_cid.exception(f"Request error occurred while fetching models for tag '{tag}': {e}")
                    print("A connection error occurred. Check internet and try again.")
                    failed_search_requests.append(base_url)
                    break  # Exit loop on error
    return model_ids


async def download_images_for_model_with_tag_check(model_ids, option_folder, timeout_value, quality='SD', tag_to_check=None, tag_dir_name=None, sanitized_tag_dir_name=None, disable_prompt_check=False, allow_redownload=2):
    global NEW_IMAGES_DOWNLOADED, download_stats
    failed_urls = []
    images_without_meta = 0
    total_api_items = 0
    total_downloaded_items = 0

    if tag_to_check is None:
        tag_to_check = tag_dir_name

    for model_id in model_ids:
        url = f"{BASE_URL}?modelId={str(model_id)}&nsfw=X"
        current_visited_pages = set()

        while url:
            if url in current_visited_pages:
                logger_cid.info(f"URL {url} already visited. Ending loop.")
                break
            current_visited_pages.add(url)

            async with httpx.AsyncClient() as client:
                async with semaphore:
                    try:
                        response = await client.get(url, timeout=timeout_value, headers={"User-Agent": "Mozilla/5.0"})
                        response.raise_for_status()

                        data = response.json()
                        items = data.get('items', [])
                        total_api_items += len(items)

                        # Model directory
                        model_name = "unknown_model"
                        if items and isinstance(items[0], dict):
                            model_name = items[0]["meta"].get("Model", "unknown_model") if isinstance(items[0].get("meta"), dict) else "unknown_model"
                        tag_dir = option_folder / sanitized_tag_dir_name
                        tag_dir.mkdir(parents=True, exist_ok=True)

                        with tag_model_mapping_lock:
                            if tag_dir_name not in tag_model_mapping:
                                tag_model_mapping[tag_dir_name] = []
                            tag_model_mapping[tag_dir_name].append((model_id, model_name))

                        model_dir = tag_dir / f"model_{(model_id)}"
                        model_dir.mkdir(parents=True, exist_ok=True)

                        # Process in smaller batches
                        for item in items:
                            item_meta = item.get("meta")
                            prompt = item_meta.get("prompt", "").replace(" ", "_") if isinstance(item_meta, dict) else ""

                            if disable_prompt_check or (tag_to_check and all(word in prompt.lower() for word in tag_to_check.lower().split("_"))):
                                image_id = item['id']
                                image_url = item['url']
                                image_extension = ".png" if quality == 'HD' else ".jpeg"
                                image_path = model_dir / f"{image_id}{image_extension}"

                                if allow_redownload == 2 and check_if_image_downloaded(str(image_id), image_path, quality):
                                    continue

                                # Await the download and metadata writing
                                download_success, reason = await download_image(image_url, image_path, timeout_value, quality)

                                if download_success:
                                    NEW_IMAGES_DOWNLOADED = True
                                    total_downloaded_items += 1
                                    tags = [tag_to_check] if tag_to_check else []
                                    mark_image_as_downloaded(str(image_id), image_path, quality, tags=tags, url=item['url'])
                                    download_stats["downloaded"].append(str(image_path))
                                else:
                                    logger_cid.error(f"Failed to download image {image_id}: {reason}")
                                    download_stats["skipped"].append((item['url'], reason))

                                meta_output_path = model_dir / f"{image_id}_meta.txt"
                                await write_meta_data(item.get("meta"), meta_output_path, image_id, item.get('username', 'unknown'))
                                if not item.get("meta"):
                                    images_without_meta += 1
                            else:
                                logger_cid.info(f"Skipping image {item['id']} as it doesn't pass the prompt check")

                        sort_images_by_model_name(model_dir)
                        metadata = data['metadata']
                        next_page = metadata.get('nextPage')
                        if next_page:
                            url = next_page
                            await asyncio.sleep(3)  # Delay between requests
                        else:
                            break
                    except Exception as e:
                        logger_cid.error(f"Error processing URL {url}: {str(e)}")
                        failed_urls.append(url)
                        continue # Continue to the next URL

    return [], failed_urls, images_without_meta, sanitized_tag_dir_name, total_api_items, total_downloaded_items # Tasks are not used anymore.

def sort_images_by_tag(option_folder, tag_model_mapping):
    with tag_model_mapping_lock:
        for tag, model_ids in tag_model_mapping.items():
            sanitized_tag = tag.replace(" ", "_")
            tag_dir = option_folder / sanitized_tag
            if not any(tag_dir.iterdir()):
                print(f"No images found for the tag: {tag}")

def write_summary_to_csv(tag, downloaded_images, option_folder, tag_model_mapping):
    with tag_model_mapping_lock:
        tag_dir = option_folder / tag.replace(" ", "_")
        for model_info in tag_model_mapping.get(tag, []):
            model_id, model_name = model_info
            model_dir = tag_dir / f"model_{model_id}"

            if not model_dir.exists():
                print(f"The {model_dir} directory does not exist. Skipping CSV creation.")
                continue
            csv_file = model_dir / f"{tag.replace(' ', '_')}_summary_{datetime.now().strftime('%Y%m%d')}.csv"
            with open(csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Current Tag", "Previously Downloaded Tag", "Image Path", "Download URL"])
                for image_id, image_info in downloaded_images.items():
                    if tag in image_info.get("tags", []):
                        for prev_tag in image_info.get("tags", []):
                            if prev_tag != tag:
                                relative_path = Path(image_info["path"]).relative_to(model_dir)
                                writer.writerow([tag, prev_tag, relative_path, image_info["url"]])

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

    if identifier_type in ['model', 'modelVersion']:
        dir_name = option_folder / f"{identifier_type}_{identifier}"
    elif identifier_type == 'username':
        dir_name = option_folder / identifier.strip()
    else:
        logger_cid.error(f"Invalid identifier_type: {identifier_type}")
        return [], 0, 0, 0

    dir_name.mkdir(parents=True, exist_ok=True)
    current_visited_pages = set() # Per identifier

    while url:
        if url in current_visited_pages:
            logger_cid.info(f"URL {url} already visited. Ending loop.")
            break
        current_visited_pages.add(url)
        async with httpx.AsyncClient() as client:
            async with semaphore:
                try:
                    response = await client.get(url, timeout=timeout_value,headers={"User-Agent": "Mozilla/5.0"})
                    response.raise_for_status()
                    data = response.json()
                    items = data.get('items', [])
                    logger_cid.info(f"Received {len(items)} items from API for {identifier_type} {identifier}")
                    total_items += len(items)

                    for item in items:
                        image_id = item['id']
                        image_url = item['url']
                        image_extension = ".png" if quality == 'HD' else ".jpeg"
                        image_path = dir_name / f"{image_id}{image_extension}"


                        if allow_redownload == 2 and check_if_image_downloaded(str(image_id), image_path, quality):
                            continue

                        download_success, reason = await download_image(image_url, image_path, timeout_value, quality)
                        if download_success:
                            NEW_IMAGES_DOWNLOADED = True
                            total_downloaded += 1
                            mark_image_as_downloaded(str(image_id), image_path, quality)
                            download_stats["downloaded"].append(str(image_path)) # String
                        else:
                            download_stats["skipped"].append((item['url'], reason))

                        meta_output_path = dir_name / f"{image_id}_meta.txt"
                        await write_meta_data(item.get("meta"), meta_output_path, image_id, item.get('username', 'unknown'))
                        if not item.get("meta"):
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

# --- Input Handling Functions ---

def check_mismatched_arguments():
    if args.mode:
        if args.mode == 1 and (args.model_id or args.model_version_id or args.tags):
            print("Warning: --model_id, --model_version_id, and --tags are not used in username mode. These arguments will be ignored.")
            logger_cid.warning("--model_id, --model_version_id, and --tags are not used in username mode.")
        elif args.mode == 2 and (args.username or args.model_version_id or args.tags):
            print("Warning: --username, --model_version_id, and --tags are not used in model ID mode. These arguments will be ignored.")
            logger_cid.warning("--username, --model_version_id, and --tags are not used in model ID mode.")
        elif args.mode == 3 and (args.username or args.model_id or args.model_version_id):
            print("Warning: --username, --model_id, and --model_version_id are not used in tag search mode. These arguments will be ignored.")
            logger_cid.warning("--username, --model_id, and --model_version_id are not used in tag search mode.")
        elif args.mode == 4 and (args.username or args.model_id or args.tags):
            print("Warning: --username, --model_id, and --tags are not used in model version ID mode. These arguments will be ignored.")
            logger_cid.warning("--username, --model_id, and --tags are not used in model version ID mode.")

def get_timeout_value():
    if args.timeout:
        return args.timeout
    else:
        timeout_input = input(f"Enter timeout value (in seconds, default {TIMEOUT}): ")
        return int(timeout_input) if timeout_input.isdigit() and int(timeout_input) > 0 else TIMEOUT

def get_quality():
    if args.quality:
        return 'HD' if args.quality == 2 else 'SD'
    else:
        quality_choice = input(f"Choose image quality (1 for SD, 2 for HD, default {1 if DEFAULT_QUALITY == 'SD' else 2}): ")
        return 'HD' if quality_choice == '2' else 'SD'

def get_redownload_option():
    if args.redownload:
        return args.redownload
    else:
        allow_redownload_choice = input(f"Allow re-downloading of images (1 for Yes, 2 for No) [default: 2]: ")
        return 1 if allow_redownload_choice == '1' else 2

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
                logger_cid.warning("Invalid input. Enter numeric model version IDs.")

# --- Main Function ---

async def main():
    global downloaded_images, args
    downloaded_images = load_downloaded_images()

    if is_command_line_mode():
        print("Running in command-line mode.")
        logger_cid.info("Running in command-line mode")
    else:
        print("Running in interactive mode.")
        logger_cid.info("Running in interactive mode")

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
        option_folder = create_option_folder('Username_Search', OUTPUT_DIR)
        for username in usernames:
            f_urls, i_without_meta, t_items, t_downloaded = await download_images(username.strip(), option_folder, 'username', timeout_value, quality, allow_redownload)
            failed_urls.extend(f_urls)
            images_without_meta += i_without_meta
            total_api_items += t_items
            total_downloaded_items += t_downloaded

    elif choice == "2":
        model_ids = get_model_ids()
        option_folder = create_option_folder('Model_ID_Search', OUTPUT_DIR)
        for model_id in model_ids:
            f_urls, i_without_meta, t_items, t_downloaded = await download_images(model_id.strip(), option_folder, 'model', timeout_value, quality, allow_redownload)
            failed_urls.extend(f_urls)
            images_without_meta += i_without_meta
            total_api_items += t_items
            total_downloaded_items += t_downloaded

    elif choice == "3":
        tags = get_tags()
        disable_prompt_check = get_disable_prompt_check()
        option_folder = create_option_folder('Model_Tag_Search', OUTPUT_DIR)

        for tag in tags:
            sanitized_tag_dir_name = tag.replace(" ", "_")
            model_ids = await search_models_by_tag(tag.replace("_", "%20"), failed_search_requests)
            tag_to_check = None if disable_prompt_check else tag
            _, f_urls, i_without_meta, _, api_items, downloaded_items = await download_images_for_model_with_tag_check(model_ids, option_folder, timeout_value, quality, tag_to_check, tag, sanitized_tag_dir_name, disable_prompt_check, allow_redownload)
            failed_urls.extend(f_urls)
            images_without_meta += i_without_meta
            total_api_items += api_items
            total_downloaded_items += downloaded_items

        sort_images_by_tag(option_folder, tag_model_mapping)
        for tag_key, model_ids_list in tag_model_mapping.items():
            write_summary_to_csv(tag_key, downloaded_images, option_folder, tag_model_mapping)

    elif choice == "4":
        model_version_ids = get_model_version_ids()
        option_folder = create_option_folder('Model_Version_ID_Search', OUTPUT_DIR)
        for model_version_id in model_version_ids:
            f_urls, i_without_meta, t_items, t_downloaded = await download_images(model_version_id.strip(), option_folder, 'modelVersion', timeout_value, quality, allow_redownload)
            failed_urls.extend(f_urls)
            images_without_meta += i_without_meta
            total_api_items += t_items
            total_downloaded_items += t_downloaded
    else:
        logger_cid.error("Invalid choice!")
        return

    if failed_urls:
        logger_cid.info("Retrying failed URLs...")
        for url in failed_urls: # Simple Retry
           await download_image(url, Path("."), timeout_value=timeout_value, quality=quality) # Redownload to current dir

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

    if failed_identifiers:
       logger_cid.warning("Failed identifiers:")
       for id_type, id_value in failed_identifiers:
           logger_cid.warning(f"{id_type}: {id_value}")

    logger_cid.info("Image download completed.")


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main())
