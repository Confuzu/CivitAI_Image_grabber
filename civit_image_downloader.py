import httpx
import os
import asyncio
import json
from tqdm import tqdm

# Prompt user for usernames
usernames = input("Enter usernames (comma-separated): ").split(",")

# Prompt user for timeout value
timeout_input = input("Enter timeout value (in seconds): ")
timeout = int(timeout_input) if timeout_input.isdigit() else 10

# API endpoint for retrieving image URLs
base_url = "https://civitai.com/api/v1/images"

# Directory for image downloads
output_dir = "image_downloads"
os.makedirs(output_dir, exist_ok=True)

# Function to download an image from the provided URL
async def download_image(url, output_path):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=timeout)
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
        with open(output_path, "w") as file:
            file.write(f"No meta data available for this image.\nURL: {url}\n")
    else:
        with open(output_path, "w") as file:
            for key, value in meta.items():
                file.write(f"{key}: {value}\n")


# Async function to download images for a given username
async def download_images_for_username(username):
    url = f"{base_url}?username={username.strip()}"
    failed_urls = []
    images_without_meta = 0

    while url:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=timeout)
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
                        tasks.append(download_image(image_url, image_path))

                        meta_output_path = os.path.join(user_dir, f"{image_id}_meta.txt")
                        await write_meta_data(item.get("meta"), meta_output_path, image_id, username)

                        if not item.get("meta"):
                            images_without_meta += 1

                    await asyncio.gather(*tasks)

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
    tasks = [download_images_for_username(username.strip()) for username in usernames]
    results = await asyncio.gather(*tasks)
    failed_urls = [url for sublist, _ in results for url in sublist]
    images_without_meta = sum([count for _, count in results])

    if failed_urls:
        print("Retrying failed URLs...")
        for url, username, image_id in failed_urls:
            user_dir = os.path.join(output_dir, username)
            os.makedirs(user_dir, exist_ok=True)
            output_path = os.path.join(user_dir, f"{image_id}.jpeg")
            await download_image(url, output_path)

    if images_without_meta > 0:
        print(f"{images_without_meta} images have no meta data.")

asyncio.run(main())
print("Image download completed successfully.")
