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


#logging only for debugging not productive 
log_file_path = "civit_image_downloader_log_0.7.txt"
logging.basicConfig(filename=log_file_path, level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


##########################################
# CivitAi API is fixed!#
# civit_image_downloader_0.7
##########################################

# API endpoint for retrieving image URLs
base_url = "https://civitai.com/api/v1/images"

# Directory for image downloads
output_dir = "image_downloads"
os.makedirs(output_dir, exist_ok=True)

semaphore = asyncio.Semaphore(10)

download_stats = {
    "downloaded": [],
    "skipped": [],
}

# Function to download an image from the provided URL
async def download_image(url, output_path, timeout_value, quality='SD'):
    global download_stats  # Make sure that download_stats is available globally.
    file_extension = ".png" if quality == 'HD' else ".jpeg"
    output_path_with_extension = re.sub(r'\.jpeg|\.png', file_extension, output_path, flags=re.IGNORECASE)
    if quality == 'HD':
        url = re.sub(r"width=\d{3,4}", "original=true", url)
    
    async with httpx.AsyncClient() as client:
        async with semaphore:
            try:
                
                response = await client.get(url, timeout=timeout_value)
                response.raise_for_status()  # Ensures that the HTTP request was successful
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {output_path_with_extension}")

                with open(output_path_with_extension, "wb") as file:
                    for chunk in response.iter_bytes():
                        progress_bar.update(len(chunk))
                        file.write(chunk)
                progress_bar.close()
                download_stats["downloaded"].append(output_path_with_extension)  # Recording as successfully downloaded
                return True
            except Exception as e:
                reason = str(e)
                if isinstance(e, httpx.RequestError) or isinstance(e, httpx.ConnectError):
                    reason = "Network error while downloading the image. Please check your internet connection and try again"
                elif isinstance(e, httpx.HTTPStatusError):
                    reason = f"Error downloading the image. Server response: {e.response.status_code} Please try again later."
                elif isinstance(e, ConnectionResetError):
                    reason = "The connection to the server was closed unexpectedly. This could be a temporary network problem. Please try again later"
                download_stats["skipped"].append((url, reason))  # Recording of the skip reason
                print(f"An unexpected error has occurred. {reason}")
                return False


# Async function to write meta data to a text file. If no meta data is available, the url to the image is written to the txt file
async def write_meta_data(meta, output_path, image_id, username):
    if not meta:
        output_path = output_path.replace(".txt", "_no_meta.txt")
        url = f"https://civitai.com/images/{image_id}?period=AllTime&periodMode=published&sort=Newest&view=feed&username={username}&withTags=false"
        with open(output_path, "w", encoding='utf-8') as file:
            file.write(f"No meta data available for this image.\nURL: {url}\n")
    else:
        with open(output_path, "w", encoding='utf-8') as file:
            for key, value in meta.items():
                file.write(f"{key}: {value}\n")


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

    # Merge tags using a generator expression
        merged_tags = list(set(downloaded_images.get(image_key, {}).get("tags", [])) | set(tags or []))
        
    # Create a new entry for the image
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


NEW_IMAGES_DOWNLOADED = False

def manual_copy(src, dst):
    global SOURCE_MISSING_MESSAGE_SHOWN
    # Check if the source file exists
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst)  # Copy file preserving file metadata
        except Exception as e:
            if NEW_IMAGES_DOWNLOADED:
                print(f"Failed to copy file {src} to {dst}. Error: {e}")
            elif not SOURCE_MISSING_MESSAGE_SHOWN:
                print(f"Source file {src} does not exist. Skipping copy.")
                SOURCE_MISSING_MESSAGE_SHOWN = True


##Remove images and metadata files from the source directory.
def clear_source_directory(model_dir):
    files_to_remove = [f for f in os.listdir(model_dir) if f.endswith(('.jpeg', '.png')) or f.endswith('_meta.txt')]
    for file in files_to_remove:
        file_path = os.path.join(model_dir, file)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove file {file_path}. Error: {e}")


def shorten_path_name(path, max_length=120):
    if len(path) <= max_length:
        return path
    
    # Divide the path into directory and file name
    dir_name, file_name = os.path.split(path)
    
    # Shorten the file name if necessary
    if len(file_name) > max_length:
        file_name = file_name[:max_length - 3] + "..."
    
    # Check whether the directory is too long
    if len(dir_name) > max_length:
        # Shorten the directory if necessary
        dir_name = dir_name[:max_length - len(file_name) - 3] + "..."
    
    return os.path.join(dir_name, file_name)


def sort_images_by_model_name(model_dir):
    if os.path.exists(model_dir) and os.listdir(model_dir):
        # List all meta files in the directory
        meta_files = [f for f in os.listdir(model_dir) if f.endswith('_meta.txt')]

        unknown_meta_files = [f for f in meta_files if "no_meta" in f]
        known_meta_files = list(set(meta_files) - set(unknown_meta_files))

        for meta_file in known_meta_files:
            with open(os.path.join(model_dir, meta_file), 'r') as file:
                lines = file.readlines()
                model_name = None

                # Extract model name from metadata
                for line in lines:
                    if "Model:" in line:
                        model_name = line.split(":")[1].strip()
                        break
                    
                # If model name is found, copy the image and its metadata to a subdirectory
                if model_name:
                    target_dir = os.path.join(model_dir, model_name)
                    target_dir = shorten_path_name(target_dir)  # Shorten the path name
                    os.makedirs(target_dir, exist_ok=True)

                    # Check whether the file with .jpeg or .png exists
                    base_image_name = meta_file.replace('_meta.txt', '')
                    jpeg_image_path = os.path.join(model_dir, base_image_name + '.jpeg')
                    png_image_path = os.path.join(model_dir, base_image_name + '.png')
                    
                    if os.path.exists(jpeg_image_path):
                        image_name = base_image_name + '.jpeg'
                    elif os.path.exists(png_image_path):
                        image_name = base_image_name + '.png'
                    else:
                        print(f"No image file found for: {meta_file}")
                        continue  # If no image file was found, continue with the next metafile
                    
                    manual_copy(os.path.join(model_dir, image_name), os.path.join(target_dir, image_name))

                    # Copy the metadata file
                    manual_copy(os.path.join(model_dir, meta_file), os.path.join(target_dir, meta_file))

        # Move 'no meta' files and their images to 'unknown_meta' directory
        unknown_meta_dir = os.path.join(model_dir, 'unknown_meta')
        os.makedirs(unknown_meta_dir, exist_ok=True)
        for meta_file in unknown_meta_files:
            # Copy the 'no meta' file
            manual_copy(os.path.join(model_dir, meta_file), os.path.join(unknown_meta_dir, meta_file))

            # Copy the associated image (correcting the image file name)
            image_name = meta_file.replace('_meta_no_meta.txt', '.jpeg')
            manual_copy(os.path.join(model_dir, image_name), os.path.join(unknown_meta_dir, image_name))

        #Clear the source directory
        clear_source_directory(model_dir)
    else:
        print(f"No images found in {model_dir}. Skipping sorting.")    


visited_pages = set()


async def search_models_by_tag(tag, failed_search_requests=[]):
    base_url = f"https://civitai.com/api/v1/models?tag={tag}"
    model_id = set()
    async with httpx.AsyncClient() as client:
        async with semaphore:
            while base_url:
                try:
                    response = await client.get(base_url)
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
                        logger.error(f"Server response: {response.status_code} for URL: {base_url}")
                        print(f"Unexpected server response: {response.status_code}. Please try again later.")
                        failed_search_requests.append(base_url)
                        break
                except httpx.RequestError as e:
                    logger.exception(f"Request error occurred while fetching models for tag '{tag}': {e}")
                    print("A connection error occurred. Please check your internet connection and try again.")
                    failed_search_requests.append(base_url)
                    break
        return model_id


tag_model_mapping = {}

async def download_images_for_model_with_tag_check(model_ids, timeout_value, quality='SD', tag_to_check=None, tag_dir_name=None, sanitized_tag_dir_name=None, disable_prompt_check=False):
    global NEW_IMAGES_DOWNLOADED
    failed_urls = []
    images_without_meta = 0
    tasks = []

    if tag_to_check is None:
        tag_to_check = tag_dir_name

    for model_id in model_ids:
        url = f"{base_url}?modelId={str(model_id)}"
        visited_pages = set()  # Reset the visited_pages set for each model

        while url:
            if url in visited_pages:
                print(f"URL {url} already visited. Ending loop.")
                break
            visited_pages.add(url)
            async with httpx.AsyncClient() as client:
                async with semaphore:
                    try:
                        response = await client.get(url, timeout=timeout_value)
                        if response.status_code == 200:
                            try:
                                data = response.json()
                            except json.JSONDecodeError as e:
                                print(f"Invalid JSON data: {str(e)}")
                                break
                            items = data.get('items', [])
                            # Create directory for model ID
                            model_name = "unknown_model"
                            if items and isinstance(items[0], dict):
                                model_name = items[0]["meta"].get("Model", "unknown_model") if isinstance(items[0].get("meta"), dict) else "unknown_model"

                            tag_dir = os.path.join(output_dir, sanitized_tag_dir_name)
                            os.makedirs(tag_dir, exist_ok=True)

                            with tag_model_mapping_lock:
                                if tag_dir_name not in tag_model_mapping:
                                    tag_model_mapping[tag_dir_name] = []
                                tag_model_mapping[tag_dir_name].append((model_id, model_name))
                            
                            model_dir = os.path.join(tag_dir, f"model_{model_id}")
                            os.makedirs(model_dir, exist_ok=True)


                            for item in items:
                                item_meta = item.get("meta") if item else None
                                prompt = item_meta.get("prompt", "").replace(" ", "_") if isinstance(item_meta, dict) else ""
                                if disable_prompt_check:
                                    # Download the image (no prompt check)
                                    image_id = item['id']
                                    image_url = item['url']
                                    image_extension = ".png" if quality == 'HD' else ".jpeg"
                                    image_path = os.path.join(model_dir, f"{image_id}{image_extension}")

                                    # Check if the image has already been downloaded
                                    if not check_if_image_downloaded(str(image_id), image_path, quality):
                                        task = download_image(image_url, image_path, timeout_value, quality)
                                        tasks.append(task)
                                else:
                                    tag_words = tag_to_check.lower().split("_")
                                    if all(word in prompt.lower() for word in tag_words):
                                        # Download the image (prompt check passed)
                                        image_id = item['id']
                                        image_url = item['url']
                                        image_extension = ".png" if quality == 'HD' else ".jpeg"
                                        image_path = os.path.join(model_dir, f"{image_id}{image_extension}")
                                
                                        # Check if the image has already been downloaded
                                        if not check_if_image_downloaded(str(image_id), image_path, quality):
                                            task = download_image(image_url, image_path, timeout_value, quality)
                                            tasks.append(task)
                                    else:
                                        continue

                            # This will run all download tasks asynchronously
                            download_results = await asyncio.gather(*tasks)
                            tasks = []  # Reset the tasks list after gathering the results

                            # Check the results and mark the downloaded images
                            for idx, download_success in enumerate(download_results):
                                image_id = items[idx]['id']
                                if download_success:
                                    NEW_IMAGES_DOWNLOADED = True
                                    image_path = os.path.join(model_dir, f"{image_id}{'.png' if quality == 'HD' else '.jpeg'}")
                                    tags = [tag_to_check] if tag_to_check else []
                                    mark_image_as_downloaded(str(image_id), image_path, quality, tags=tags, url=image_url)
                                else:
                                    logger.error(f"Failed to download image {image_id}")

                                meta_output_path = os.path.join(model_dir, f"{image_id}_meta.txt")
                                await write_meta_data(items[idx].get("meta"), meta_output_path, image_id, items[idx].get('username', 'unknown'))
                                if not items[idx].get("meta"):
                                    images_without_meta += 1
                            sort_images_by_model_name(model_dir)
                            metadata = data['metadata']
                            next_page = metadata.get('nextPage')
                            if next_page:
                                url = next_page
                                await asyncio.sleep(3)  # Add a delay between requests
                            else:
                                break
                    except httpx.TimeoutException as e:
                        print(f"Request timed out: {str(e)}")
                        failed_urls.append(url)  # Append the failed URL to the list
                        logging.info(f"URL hinzugefügt zu failed_urls: {url}")
                        continue
                    except httpx.HTTPError as e:
                        print(f"HTTP error occurred: {str(e)}")
                        failed_urls.append(url)  # Append the failed URL to the list
                        logging.info(f"URL hinzugefügt zu failed_urls: {url}")
                        continue
                    except Exception as e:
                        print(f"An error occurred: {str(e)}")
                        failed_urls.append(url)  # Append the failed URL to the list
                        logging.info(f"URL hinzugefügt zu failed_urls: {url}")
                        continue

    return tasks, failed_urls, images_without_meta, sanitized_tag_dir_name


def sort_images_by_tag(tag_model_mapping):
    with tag_model_mapping_lock:
        for tag, model_ids in tag_model_mapping.items():
            sanitized_tag = tag.replace(" ", "_")
            tag_dir = os.path.join(output_dir, sanitized_tag)
            if not os.listdir(tag_dir):
                print(f"No images found for the tag: {tag}")


def write_summary_to_csv(tag, downloaded_images, tag_model_mapping):
    with tag_model_mapping_lock:
        tag_dir = os.path.join(output_dir, tag.replace(" ", "_"))
        for model_info in tag_model_mapping.get(tag, []):
            model_id, model_name = model_info
            model_dir = os.path.join(tag_dir, f"model_{model_id}")
        
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


async def download_images(identifier, identifier_type, timeout_value, quality='SD'):
    global NEW_IMAGES_DOWNLOADED
    if identifier_type == 'model':
        url = f"{base_url}?modelId={str(identifier)}"
    elif identifier_type == 'username':
        url = f"{base_url}?username={identifier.strip()}"
    else:
        raise ValueError("Invalid identifier_type. Should be 'model' or 'username'.")

    failed_urls = []
    images_without_meta = 0

    while url:
        if url in visited_pages:
            print(f"URL {url} already visited. Ending loop.")
            break
        visited_pages.add(url)     
        async with httpx.AsyncClient() as client:
            async with semaphore:
                try:
                    response = await client.get(url, timeout=timeout_value)
                    if response.status_code == 200:
                        try:
                            data = response.json()
                        except json.JSONDecodeError as e:
                            print(f"Invalid JSON data: {str(e)}")
                            break

                        items = data.get('items', [])

                        # Create directory for model ID or username
                        if identifier_type == 'model':
                            model_name = "unknown_model"
                            if items and isinstance(items[0], dict):
                                model_name = items[0]["meta"].get("Model", "unknown_model") if isinstance(items[0].get("meta"), dict) else "unknown_model"
                            dir_name = os.path.join(output_dir, f"model_{identifier}")
                        elif identifier_type == 'username':
                            dir_name = os.path.join(output_dir, identifier.strip())

                        os.makedirs(dir_name, exist_ok=True)

                        tasks = []
                        for item in items:
                            image_id = item['id']
                            image_url = item['url']
                            image_extension = ".png" if quality == 'HD' else ".jpeg"
                            image_path = os.path.join(dir_name, f"{image_id}{image_extension}")

                            # Check if the image has already been downloaded
                            if not check_if_image_downloaded(str(image_id), image_path, quality):
                                task = download_image(image_url, image_path, timeout_value, quality)
                                tasks.append(task)

                        # This will run all download tasks asynchronously
                        download_results = await asyncio.gather(*tasks)

                        # Check the results and mark the downloaded images
                        for idx, download_success in enumerate(download_results):
                            image_id = items[idx]['id']
                            if download_success:
                                NEW_IMAGES_DOWNLOADED = True
                                image_path = os.path.join(dir_name, f"{image_id}{'.png' if quality == 'HD' else '.jpeg'}")
                                mark_image_as_downloaded(str(image_id), image_path, quality)
                            
                            meta_output_path = os.path.join(dir_name, f"{image_id}_meta.txt")
                            await write_meta_data(items[idx].get("meta"), meta_output_path, image_id, items[idx].get('username', 'unknown'))
                            if not items[idx].get("meta"):
                                images_without_meta += 1
                        
                        metadata = data['metadata']
                        next_page = metadata.get('nextPage')

                        if next_page:
                            url = next_page
                            await asyncio.sleep(3)  # Add a delay between requests
                        else:
                            break
                except httpx.TimeoutException as e:
                    print(f"Request timed out: {str(e)}")
                    continue

    if identifier_type == 'model':
        # Sorting the images by model name after downloading them
        sort_images_by_model_name(dir_name)
    elif identifier_type == 'username':
        sort_images_by_model_name(dir_name)

    return failed_urls, images_without_meta


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
    try:
        global downloaded_images
        downloaded_images = load_downloaded_images()

        # Universal user inputs for timeout and quality
        timeout_input = input("Enter timeout value (in seconds): ")
        if timeout_input.isdigit() and int(timeout_input) > 0:
            timeout_value = int(timeout_input)
        else:
            print("Invalid timeout value. Using default value of 20 seconds.")
            timeout_value = 20

        quality_choice = input("Choose image quality (1 for SD, 2 for HD): ")
        if quality_choice == '2':
            quality = 'HD'
        elif quality_choice == '1':
            quality = 'SD'
        else:
            print("Invalid quality choice. Using default quality SD.")
            quality = 'SD'

        failed_search_requests = []
        failed_urls = []
        images_without_meta = 0

        # Extended choice options including tag search
        choice = input("Choose mode (1 for username, 2 for model ID, 3 for tag search): ")
        tasks = []

        if choice == "3":
            tags_input = input("Enter tags (comma-separated): ")
            tags = [tag.strip().replace(" ", "_") for tag in tags_input.split(',')]
            disable_prompt_check = input("Disable prompt check? (y/n): ").lower() in ['y', 'yes']

            for tag in tags:
                sanitized_tag_dir_name = tag.replace(" ", "_")
                model_ids = await search_models_by_tag(tag.replace("_", "%20"), failed_search_requests)
                tag_to_check = None if disable_prompt_check else tag
                tasks_for_tag, failed_urls_for_tag, images_without_meta_for_tag, sanitized_tag_dir_name_for_tag = await download_images_for_model_with_tag_check(model_ids, timeout_value, quality, tag_to_check, tag, sanitized_tag_dir_name, disable_prompt_check)
                tasks.extend(tasks_for_tag)
                failed_urls.extend(failed_urls_for_tag)
                images_without_meta += images_without_meta_for_tag

        elif choice == "1":
            usernames = input("Enter usernames (comma-separated): ").split(",")
            tasks.extend([download_images(username.strip(), 'username', timeout_value, quality) for username in usernames])

        elif choice == "2":
            model_ids = input("Enter model IDs (comma-separated): ").split(",")
            tasks.extend([download_images(model_id.strip(), 'model', timeout_value, quality) for model_id in model_ids])

        else:
            print("Invalid choice!")
            return

        # Execute all collected tasks
        results = await asyncio.gather(*tasks)

        # Extract failed_urls and images_without_meta from the results
        failed_urls.extend([url for result in results for url in result[0]])
        images_without_meta += sum([count for result in results for count in result[1:]])

        # Sort images into tag-related folders
        sort_images_by_tag(tag_model_mapping)

        for tag, model_ids in tag_model_mapping.items():
            write_summary_to_csv(tag, downloaded_images, tag_model_mapping)

        if failed_urls:
            print("Retrying failed URLs...")
            for url, id_or_username, image_id in failed_urls:
                dir_name = os.path.join(output_dir, id_or_username)
                os.makedirs(dir_name, exist_ok=True)
                output_path = os.path.join(dir_name, f"{image_id}{'.png' if quality == 'HD' else '.jpeg'}")
                await download_image(url, output_path)

        # Attempt to retry failed search requests
        if failed_search_requests:
            print("Retrying failed search requests...")
            for url in failed_search_requests:
                tag = url.split("tag=")[-1]
                await search_models_by_tag(tag, [])

        if images_without_meta > 0:
            print(f"{images_without_meta} images have no meta data.")

        print_download_statistics()

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        raise

asyncio.run(main())
print("Image download completed.")
