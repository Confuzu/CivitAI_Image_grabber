import httpx
import os
import asyncio
import json
from tqdm import tqdm

# Prompt user for usernames
usernames = input("Enter usernames (comma-separated): ").split(",")

# Prompt user for timeout value
timeout_input = input("Enter timeout value (in seconds): ")
timeout = int(timeout_input) if timeout_input.isdigit() else 5

# API endpoint for retrieving image URLs
base_url = "https://civitai.com/api/v1/images"

# Function to download an image from the provided URL
async def download_image(url, output_path):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=timeout)  # Set the timeout for the request
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
            with open(output_path, "wb") as file:
                async for chunk in response.aiter_bytes():
                    progress_bar.update(len(chunk))
                    file.write(chunk)
            progress_bar.close()
        except (httpx.RequestError, httpx.HTTPStatusError, asyncio.TimeoutError) as e:
            # If an error occurs, print the error message and return the URL
            print(f"Error downloading image: {e}")
            return url

# Create a directory for image downloads
output_dir = "image_downloads"
os.makedirs(output_dir, exist_ok=True)

# Define the async function to download images for a given username
async def download_images_for_username(username):
    # Format the initial URL with the username
    url = f"{base_url}?username={username.strip()}"

    failed_urls = []  # Store the failed URLs for retry

    while url:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=timeout)  # Set the timeout for the request
                if response.status_code == 200:  # Verify the response status code
                    try:
                        data = json.loads(response.text)  # Validate and load the response as JSON
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON data: {str(e)}")
                        break  # Exit the loop if invalid JSON is encountered

                    items = data['items']

                    # Create a folder for the username
                    user_dir = os.path.join(output_dir, username.strip())
                    os.makedirs(user_dir, exist_ok=True)

                    # Download images for each item
                    tasks = []
                    for item in items:
                        image_id = item['id']
                        image_url = item['url']

                        # Download the image (add the task to the list)
                        image_path = os.path.join(user_dir, f"{image_id}.jpeg")
                        tasks.append(download_image(image_url, image_path))

                    # Wait for all image downloads to complete
                    failed_results = await asyncio.gather(*tasks)

                    # Store the failed URLs for retry with the username and image ID
                    failed_urls.extend([(url, username.strip(), item['id']) for url, item in zip(failed_results, items) if url])

                    # Check if there is a next page
                    metadata = data['metadata']
                    next_page = metadata.get('nextPage')
                    if next_page:
                        url = next_page
                    else:
                        break
                else:
                    print(f"Error occurred during the request: {response.status_code}")
                    break  # Exit the loop if the request was not successful
            except httpx.TimeoutException as e:
                print(f"Request timed out: {str(e)}")
                continue  # Retry the request if a timeout occurs

    return failed_urls

# Run the async functions concurrently for all usernames
async def main():
    tasks = [download_images_for_username(username.strip()) for username in usernames]
    failed_results = await asyncio.gather(*tasks)
    failed_urls = [url for sublist in failed_results for url in sublist]

    # Retry failed URLs within the respective username folders
    if failed_urls:
        print("Retrying failed URLs...")
        for url, username, image_id in failed_urls:
            user_dir = os.path.join(output_dir, username)
            os.makedirs(user_dir, exist_ok=True)  # Create the user directory if it doesn't exist
            output_path = os.path.join(user_dir, f"{image_id}.jpeg")
            await download_image(url, output_path)

# Run the main async function
asyncio.run(main())

# Print completion message
print("Image download completed successfully.")
