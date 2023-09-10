import httpx
import os
import asyncio
import json
from tqdm import tqdm
import shutil

# API endpoint for retrieving image URLs
base_url = "https://civitai.com/api/v1/images"

# Directory for image downloads
output_dir = "image_downloads"
os.makedirs(output_dir, exist_ok=True)

# Function to download an image from the provided URL
async def download_image(url, output_path, timeout_value):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=timeout_value)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
            with open(output_path, "wb") as file:
                async for chunk in response.aiter_bytes():
                    progress_bar.update(len(chunk))
                    file.write(chunk)
            progress_bar.close()
            return True
        except (httpx.RequestError, httpx.HTTPStatusError, asyncio.TimeoutError) as e:
            # If an error occurs, print the error message and return the URL
            print(f"Error downloading image: {e}")
            return False

# Async function to write meta data to a text file
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


TRACKING_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "downloaded_images.txt")

DOWNLOADED_IMAGES = set() # Here we store the IDs of the downloaded images

def load_downloaded_images():
    global DOWNLOADED_IMAGES
    try:
        with open(TRACKING_FILE, "r") as file:
            DOWNLOADED_IMAGES = set(file.read().splitlines())
    except FileNotFoundError:
        pass  


def check_if_image_downloaded(image_id):
    return image_id in DOWNLOADED_IMAGES


def mark_image_as_downloaded(image_id):
    """Mark the image with the given ID as downloaded by adding its ID to the tracking file."""
    with open(TRACKING_FILE, "a") as file:
        file.write(image_id + "\n")

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


#Remove images and metadata files from the source directory.
def clear_source_directory(model_dir):
    files_to_remove = [f for f in os.listdir(model_dir) if f.endswith('.jpeg') or f.endswith('_meta.txt')]
    for file in files_to_remove:
        file_path = os.path.join(model_dir, file)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove file {file_path}. Error: {e}")


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
                    os.makedirs(target_dir, exist_ok=True)

                    # Copy the image
                    image_name = meta_file.replace('_meta.txt', '.jpeg')
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

        # Clear the source directory
        clear_source_directory(model_dir)
    else:
        print(f"No images found in {model_dir}. Skipping sorting.")    

# Async function to download images for a given Model ID
async def download_images_for_model(model_id, timeout_value):
    global NEW_IMAGES_DOWNLOADED
    url = f"{base_url}?modelId={str(model_id)}"
    failed_urls = []
    images_without_meta = 0

    while url:
        async with httpx.AsyncClient() as client:
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

                    model_dir = os.path.join(output_dir, f"model_{model_id}_{model_name}")
                    os.makedirs(model_dir, exist_ok=True)

                    tasks = []
                    for item in items:
                        image_id = item['id']
                        image_url = item['url']
                        image_path = os.path.join(model_dir, f"{image_id}.jpeg")
                        
                        # Check if the image has already been downloaded
                        if not check_if_image_downloaded(str(image_id)):
                            task = download_image(image_url, image_path, timeout_value)
                            tasks.append(task)

                    # This will run all download tasks asynchronously
                    download_results = await asyncio.gather(*tasks)

                    # Check the results and mark the downloaded images
                    for idx, download_success in enumerate(download_results):
                        image_id = items[idx]['id']
                        if download_success:
                            NEW_IMAGES_DOWNLOADED = True
                            mark_image_as_downloaded(str(image_id))

                        meta_output_path = os.path.join(model_dir, f"{image_id}_meta.txt")
                        await write_meta_data(item.get("meta"), meta_output_path, image_id, item.get('username', 'unknown'))

                        if not item.get("meta"):
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

    # Sorting the images by model name after downloading them
    sort_images_by_model_name(model_dir)

    return failed_urls, images_without_meta

# Async function to download images for a given username
async def download_images_for_username(username, timeout_value):
    global NEW_IMAGES_DOWNLOADED
    url = f"{base_url}?username={username.strip()}"
    failed_urls = []
    images_without_meta = 0

    while url:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=timeout_value)
                if response.status_code == 200:
                    try:
                        data = json.loads(response.text)
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON data: {str(e)}")
                        break

                    items = data['items']

                    user_dir = os.path.join(output_dir, username.strip())
                    os.makedirs(user_dir, exist_ok=True)

                    tasks = []

                    for item in items:
                        image_id = item['id']
                        image_url = item['url']
                        image_path = os.path.join(user_dir, f"{image_id}.jpeg")
                        # Check if the image has already been downloaded
                        if not check_if_image_downloaded(str(image_id)):
                            task = download_image(image_url, image_path, timeout_value)
                            tasks.append(task)

                    # This will run all download tasks asynchronously
                    download_results = await asyncio.gather(*tasks)

                    # Check the results and mark the downloaded images
                    for idx, download_success in enumerate(download_results):
                        image_id = items[idx]['id']
                        if download_success:
                            NEW_IMAGES_DOWNLOADED = True
                            mark_image_as_downloaded(str(image_id))

                        meta_output_path = os.path.join(user_dir, f"{image_id}_meta.txt")
                        await write_meta_data(item.get("meta"), meta_output_path, image_id, username)

                        if not item.get("meta"):
                            images_without_meta += 1

                    metadata = data['metadata']
                    next_page = metadata.get('nextPage')

                    if next_page:
                        url = next_page
                        await asyncio.sleep(3)  # Add a delay between requests
                    else:
                        break
                else:
                    print(f"Error occurred during the request: {response.status_code}")
                    break
            except httpx.TimeoutException as e:
                print(f"Request timed out: {str(e)}")
                continue

    return failed_urls, images_without_meta


# Main async function
async def main():
    load_downloaded_images()
    choice = input("Choose mode (1 for username, 2 for model ID): ")

# Prompt user for timeout value
    timeout_input = input("Enter timeout value (in seconds): ")
    timeout_value = int(timeout_input) if timeout_input.isdigit() else 20

    if choice == "1":
        usernames = input("Enter usernames (comma-separated): ").split(",")
        tasks = [download_images_for_username(username.strip(), timeout_value) for username in usernames]
    elif choice == "2":
        model_ids = input("Enter model IDs (comma-separated): ").split(",")
        tasks = [download_images_for_model(model_id.strip(), timeout_value) for model_id in model_ids]
    else:
        print("Invalid choice!")
        return

    

    results = await asyncio.gather(*tasks)
    failed_urls = [url for sublist, _ in results for url in sublist]
    images_without_meta = sum([count for _, count in results])

    if failed_urls:
        print("Retrying failed URLs...")
        for url, id_or_username, image_id in failed_urls:
            dir_name = os.path.join(output_dir, id_or_username)
            os.makedirs(dir_name, exist_ok=True)
            output_path = os.path.join(dir_name, f"{image_id}.jpeg")
            await download_image(url, output_path)

    if images_without_meta > 0:
        print(f"{images_without_meta} images have no meta data.")


asyncio.run(main())
print("Image download completed successfully.")
