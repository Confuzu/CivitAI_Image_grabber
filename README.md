# Civit Image grabber 2.1

It downloads all the images and Videos from a provided Username, Model ID or Model TAG from CivitAI. 
Should the API not spit out all the data for all images then I'm sorry. 
The script can only download where data is provided.

The files are downloaded into a folder with the name of the `user`, `ModelID` or the `TAG` <br /> 
Second Level is the `Model Name` with which the image was generated.<br />
Videos without model metadata are placed in a dedicated `videos/` subfolder.



# Installation


1.  **Install Python 3** Ensure you have Python 3.8 or newer installed.

```
install Python3
```
   
2.  **Install Dependencies** NEW requirements for users who already use the script
```bash
pip install -r requirements.txt
```

3.  **Migrate Existing Data**
*   If you are using the previous version with a `downloaded_images.json` file and want to continue using your data for tracking, you must use the my tool `migrate_json_to_sqlite.py` to transfer your old data to the new database.
*   Make sure both `migrate_json_to_sqlite.py` and your old `downloaded_images.json` are in the same directory.
*   Run the migration tool from your terminal:
```bash
python migrate_json_to_sqlite.py
```
*   Follow the prompts. It will create `tracking_database.sqlite` and offer to rename your old JSON file.

---

# Usage

## Interactive Mode

Run the script without any command-line arguments:
```bash
python civit_image_downloader.py
```
the script will ask you to:

1.  `Enter timeout value (seconds) [default: 60]:`
2.  `Choose image quality (1=SD, 2=HD) [default: 1]:`
3.  `Allow re-downloading tracked items? (1=Yes, 2=No) [default: 2]:`
4.  `Skip video files? (y/n) [default: n]:`
5.  `Choose mode (1=user, 2=model ID, 3=tag search, 4=model version ID):` 
6.  `Enter max concurrent downloads [default: 5]:` 
7.  *(Mode-specific prompts):*
    *   Mode 1: `Enter username(s) (, separated):`
    *   Mode 1: `Enter filter tag(s) (comma-separated, optional, press Enter to skip):` (Optional: filter images by tags)
    *   Mode 1: `Disable prompt check? (y/n) [default: n]:` (If filter tags are provided)
    *   Mode 2: `Enter model ID(s) (numeric, , separated):`
    *   Mode 3: `Enter tags (, separated):`
    *   Mode 3: `Disable prompt check? (y/n) [default: n]:` (Check if tag words must be in the image prompt)
    *   Mode 4: `Enter model version ID(s) (numeric, , separated):`
    *   Mode 4: `Enter filter tag(s) (comma-separated, optional, press Enter to skip):` (Optional: filter by tags)
    *   Mode 4: `Disable prompt check? (y/n) [default: n]:` (If filter tags are provided)

If you just hit enter it will use the Default values of that Option if it has a default value.  <br /> 
 <br /> 

## Command-Line Mode

Provide arguments directly on the command line. Unspecified arguments will use their defaults. `--mode` is required.

**Available Arguments**

*   `--timeout INT` (Default: 60)
*   `--quality {1,2}` (1=SD, 2=HD, Default: SD)
*   `--redownload {1,2}` (1=Yes, 2=No, Default: 2)
*   `--mode {1,2,3,4}` (**Required**)
*   `--tags TAGS` (Comma-separated, required for Mode 3)
*   `--disable_prompt_check {y,n}` (Default: n, works with Mode 1 (with filter_tags), Mode 3 and Mode 4)  
*   `--username USERNAMES` (Comma-separated, required for Mode 1)
*   `--model_id IDS` (Comma-separated, numeric, required for Mode 2)
*   `--model_version_id IDS` (Comma-separated, numeric, required for Mode 4)
*   `--filter_tags TAGS` (Comma-separated, optional for Mode 1 and Mode 4, filters images by tags)   
*   `--output_dir PATH` (Default: "image_downloads")
*   `--semaphore_limit INT` (Default: 5)
*   `--no_sort` (Disables model subfolder sorting, Default: False/Sorting enabled)
*   `--no_videos` (Skip video files, download images only, Default: False)
*   `--max_path INT` (Default: 240)
*   `--retries INT` (Default: 2)
*   `--max_images INT` (Limit total images downloaded, Default: unlimited) 
*   `--max_per_model INT` (Limit images per model in tag searches, Default: unlimited) 
*   `--deep_scan` (Enable deep scan for users with 50K+ images, Mode 1 only, Default: False)

## Examples

*   Download HD images for user "artist1", allowing redownloads, higher concurrency:
    ```bash
    python civit_image_downloader.py --mode 1 --username "artist1" --quality 2 --redownload 1 --semaphore_limit 10
    ```
*   Download SD images for models 123 and 456, using defaults for other options:
    ```bash
    python civit_image_downloader.py --mode 2 --model_id "123, 456"
    ```
*   Download SD images for tag "sci-fi", disabling prompt check, no redownloads:
    ```bash
    python civit_image_downloader.py --mode 3 --tags "sci-fi" --disable_prompt_check y --redownload 2
    ```
*   Download images from model version 123456, filtered by tags "anime" and "portrait":
    ```bash
    python civit_image_downloader.py --mode 4 --model_version_id "123456" --filter_tags "anime,portrait" --disable_prompt_check y
    ```
*   Download only "anime" images from a specific username (Mode 1 + tag filter):
    ```bash
    python civit_image_downloader.py --mode 1 --username "artist123" --filter_tags "anime" --max_images 50
    ```
*   Download only 100 images from user "artist1" (useful for testing or sampling):
    ```bash
    python civit_image_downloader.py --mode 1 --username "artist1" --max_images 100
    ```
*   Download images from tag search with per-model limit (balanced sampling):
    ```bash
    python civit_image_downloader.py --mode 3 --tags "landscape" --max_per_model 50 --max_images 500
    ```
*   To skip all video files:
    ```bash
    python civit_image_downloader.py --mode 1 --username "artist1" --no_videos
    ```
*   Download ALL images from a user (50K+ images) using deep scan:
    ```bash
    python civit_image_downloader.py --mode 1 --username "Artist1" --deep_scan
    ```   

## Deep Scan

CivitAI's API caps pagination at ~50,000 images per user. Users with more than 50K images will silently get incomplete downloads. Pass `--deep_scan` to run additional passes that retrieve images beyond this limit using bi-directional pagination and per-model-version queries. Only applies to Mode 1 (username search). Without this flag, the script warns when a user hits the cap.


## Mixed Mode

If only some arguments are provided (e.g., only `--mode`), the script will use the provided options and prompt the user for any missing inputs.

---

## Folder Structure

The downloaded files will be organized within the specified `--output_dir` (default: `image_downloads`). Sorting (`--no_sort` flag) affects the structure inside the identifier folder.

**With Sorting Enabled (Default)**

```
image_downloads/
├── Username_Search/
│   └── [Username]/
│       ├── [Model Name Subfolder]/  # Based on image metadata 'Model' field
│       │   ├── [ImageID].jpeg       # or .png, .webp
│       │   └── [ImageID]_meta.txt
│       ├── videos/                  # Videos without parsable model metadata (.mp4, .webm)
│       │   ├── [ImageID].mp4        # or .webm
│       │   └── [ImageID]_no_meta.txt (or _meta.txt)
│       ├── invalid_metadata/        # For images with meta but no parsable 'Model' field
│       │   ├── [ImageID].jpeg
│       │   └── [ImageID]_meta.txt
│       └── no_metadata/             # For images (non-video) with no metadata found
│           ├── [ImageID].jpeg
│           └── [ImageID]_no_meta.txt
├── Model_ID_Search/
│   └── model_[ModelID]/
│       ├── [Model Name Subfolder]/
│       │   ├── [ImageID].jpeg
│       │   └── [ImageID]_meta.txt
│       ├── videos/                  # Videos without parsable model metadata (.mp4, .webm)
│       │   ├── [ImageID].mp4
│       │   └── [ImageID]_no_meta.txt (or _meta.txt)
│       ├── invalid_metadata/
│       │   ├── [ImageID].jpeg
│       │   └── [ImageID]_meta.txt
│       └── no_metadata/
│           ├── [ImageID].jpeg
│           └── [ImageID]_no_meta.txt
│
├── Model_Version_ID_Search/
│   └── modelVersion_[VersionID]/
│       ├── [Model Name Subfolder]/
│       │   ├── [ImageID].jpeg
│       │   └── [ImageID]_meta.txt
│       ├── videos/                  # Videos without parsable model metadata (.mp4, .webm)
│       │   ├── [ImageID].mp4
│       │   └── [ImageID]_no_meta.txt (or _meta.txt)
│       ├── invalid_metadata/
│       │   ├── [ImageID].jpeg
│       │   └── [ImageID]_meta.txt
│       └── no_metadata/
│           ├── [ImageID].jpeg
│           └── [ImageID]_no_meta.txt
└── Model_Tag_Search/
    └── [Sanitized_Tag_Name]/         # e.g., sci_fi_vehicle
        ├── model_[ModelID]/          # Folder for each model found under the tag
        │   ├── [Model Name Subfolder]/
        │   │   ├── [ImageID].jpeg
        │   │   └── [ImageID]_meta.txt
        │   ├── videos/               # Videos without parsable model metadata (.mp4, .webm)
        │   │   ├── [ImageID].mp4
        │   │   └── [ImageID]_no_meta.txt (or _meta.txt)
        │   ├── invalid_metadata/
        │   │   ├── [ImageID].jpeg
        │   │   └── [ImageID]_meta.txt
        │   └── no_metadata/
        │       ├── [ImageID].jpeg
        │       └── [ImageID]_no_meta.txt
        └── summary_[Sanitized_Tag_Name]_[YYYYMMDD].csv
```

**With Sorting Disabled (`--no_sort`)**

All images, videos and metadata files for a given identifier (username, model ID, model version ID, or model ID within a tag) are placed directly within that identifier's folder, without the `[Model Name Subfolder]`, `videos/`, `invalid_metadata`, or `no_metadata` subdirectories.

```
image_downloads/
├── Username_Search/
│   └── [Username]/
│       ├── [ImageID].jpeg        # or .png, .webp
│       ├── [ImageID].mp4         # or .webm (videos)
│       ├── [ImageID]_meta.txt
│       └── [ImageID]_no_meta.txt
├── Model_ID_Search/
│   └── model_[ModelID]/
│       ├── [ImageID].jpeg
│       ├── [ImageID].mp4
│       └── ...
├── Model_Version_ID_Search/
│   └── modelVersion_[VersionID]/
│       ├── [ImageID].jpeg
│       ├── [ImageID].mp4
│       └── ...
└── Model_Tag_Search/
    └── [Sanitized_Tag_Name]/
        ├── model_[ModelID]/
        │   ├── [ImageID].jpeg
        │   ├── [ImageID].mp4
        │   ├── [ImageID]_meta.txt
        │   └── [ImageID]_no_meta.txt
        └── summary_[Sanitized_Tag_Name]_[YYYYMMDD].csv
```

---

## Tracking Database (`tracking_database.sqlite`)

This file replaces the old JSON file. It stores a record of each downloaded image/video, including its path, quality, download date, associated tags (from Mode 3), original URL, and extracted checkpoint name (from metadata). You can explore this file using tools like "DB Browser for SQLite".

**Migration Tool (`migrate_json_to_sqlite.py`)**

If you are updating from a version using `downloaded_images.json`, run this separate Python script *once* in the same directory as your JSON file *before* using the main downloader. It will read the JSON and populate the new `tracking_database.sqlite` file.

```bash
python migrate_json_to_sqlite.py
```

---



# Update History

## 2.1 Deep Scan -- 50K API Pagination Cap Bypass

**New `--deep_scan` flag** for Mode 1 (username search) that retrieves images beyond the CivitAI API's 50K pagination cap.

### The Problem

CivitAI's `/api/v1/images` endpoint uses cursor-based pagination that caps at ~50,000 items (250 pages x 200 items per page). After that, `nextPage` becomes null and pagination silently stops. For users with more than 50K images, this means a standard download only retrieves a fraction of their gallery. A user with 148,208 images would only get ~50,000 without deep scan.

### How Deep Scan Works

Deep scan uses a 5-pass strategy to retrieve images beyond the 50K cap:

| Pass | Strategy | Purpose |
|------|----------|---------|
| 1 | `sort=Newest`, `nsfw=X` | Initial pass (NSFW images, newest first) |
| 2 | `sort=Oldest`, `nsfw=X` | NSFW images from the other end |
| 3 | `sort=Newest`, no nsfw param | SFW images, newest first |
| 4 | `sort=Oldest`, no nsfw param | SFW images from the other end |
| 5 | Per-`modelVersionId` queries | Targeted gap-filling for specific models |

Each pass paginates until the API stops returning results. The SQLite tracking database prevents duplicate downloads across all passes.

Pass 5 collects `modelVersionId` values discovered during passes 1-4 and queries each one individually. This catches images that fall in the "middle" between what Newest and Oldest pagination can reach. It stops after 50 consecutive model versions yield zero new images.

### The SFW/NSFW Split

The API treats NSFW and SFW images as entirely separate pagination streams, each with its own independent 50K cap. This effectively doubles the maximum reachable images to ~200K (100K NSFW + 100K SFW) before model-version fill is needed.

### Realistic Expectations

Coverage depends on how many images the user has and how they're distributed:

| User's Total Images | Expected Coverage | Notes |
|---------------------|-------------------|-------|
| 50K - 100K | 90-100% | Bi-directional pagination usually covers everything |
| 100K - 200K | 60-95% | Depends on SFW/NSFW split and model diversity |
| 200K - 500K | 30-60% | Middle images may be unreachable via any API path |
| 500K+ | 10-30% | API limitations make full retrieval unlikely |

The 50K cap is a hard API limitation. Deep scan maximizes what's retrievable but cannot guarantee 100% for very large galleries.

### Usage

```bash
# Basic deep scan
python civit_image_downloader.py --mode 1 --username "user" --deep_scan

# Multiple users -- deep scan applies to all that hit the cap
python civit_image_downloader.py --mode 1 --username "user1,user2,user3" --deep_scan
```

Without `--deep_scan`, the script warns when a user hits the cap:
```
WARNING: username appears to have hit the 50K API pagination cap (49863 items).
         Use --deep_scan to retrieve additional images beyond this limit.
```

## 2.0 New Features

**Video Download Support** 

- The script now detects and downloads video files automatically — no configuration required. 
- Detects `type: "video"` items in API responses and downloads them as `.mp4` or `.webm`
- Videos are routed to a dedicated `videos/` subfolder inside the identifier folder (all modes)

**Skip videos (`--no_videos`):**

Users who only want images can pass `--no_videos` to skip all video files returned by the API. Skipped videos appear in the end-of-run skip summary.

```bash
python civit_image_downloader.py --mode 1 --username "artist1" --no_videos
```
in the Interactive Mode it will  ask you 
4.  `Skip video files? (y/n) [default: n]:`

## 1.9 Security Update 

**Fix 1 — SSRF protection (_validate_next_page()):**

  - Any nextPage URL from the API now passes three checks before being followed: must be https://, must be ``` civitai.com or www.civitai.com,``` must start with /api/                                                                      
  - Applied to both pagination loops (_run_paginated_download and _search_models_by_tag)                                                                                                                                              
  - Invalid URLs log a warning and stop pagination gracefully rather than raising

 **Fix 2 — URL encoding (quote()):**
 
  - username in Mode 1: quote(ident, safe='') — handles any special character a CivitAI username could contain                                                                                                                        
  - Tag in _search_models_by_tag: replaces the partial .replace(" ", "%20") with proper RFC 3986 encoding       

## 1.8 Username Search with Tag Filtering  <br />

Mode 1 now supports `--filter_tags`, allowing you to download only images from a specific user that match certain tags. Previously, username search and tag filtering were completely separate modes.

**New Interactive Prompts for Mode 1:**
```
Enter filter tag(s) (comma-separated, optional, press Enter to skip):
Disable prompt check? (y/n) [default: n]:
```

**Example Scenarios:**
 
- Download only "anime" images from a specific artist
```bash
python civit_image_downloader.py --mode 1 --username "artist123" --filter_tags "anime" --max_images 50
```
- Multiple tags (image must match ALL tags)
```bash

python civit_image_downloader.py --mode 1 --username "artist123" --filter_tags "woman,photorealistic" --max_images 30
```
- Disable prompt check for faster (less strict) filtering
```bash
python civit_image_downloader.py --mode 1 --username "photographer_xyz" --filter_tags "portrait" --disable_prompt_check y
```
## 1.7 New Feature Target-Specific Database Clearing  <br /> 

The `--redownload 1` option has been enhanced to support **selective clearing** of database history. When re-downloading, the script now automatically clears only the specific target being processed, leaving all other download records untouched.<br />
- Each download is now tagged with its target type and value (`username:artist1`, `model:12345`, etc.)
- Database migration automatically adds target tracking columns on first run (non-destructive, preserves all existing data)
- When `--redownload 1` is used, the script clears only the current target's records before redownloading. <br />

**Example Scenarios:**

- Scenario 1: Re-download one user
```bash
python civit_image_downloader.py --mode 1 --username "artist1" --redownload 1 --quality 1
```
 Output: Cleared 150 previous downloads for username: artist1
 Result: Only artist1's records cleared, other users untouched


- Scenario 2: Multiple users in one run
```bash
python civit_image_downloader.py --mode 1 --username "Bob,Alice,Charlie" --redownload 1 --quality 1
```
 Result: Bob cleared → Bob downloaded → Alice cleared → Alice downloaded → Charlie cleared → Charlie downloaded


- Scenario 3: Normal download (no clearing)
```bash
python civit_image_downloader.py --mode 1 --username "artist1" --redownload 2 --quality 1
```
 Result: No clearing, skips already-downloaded images (default behavior unchanged)


- **Database Schema:**
- Added `target_type` column (stores: `username`, `model`, `tag`, `modelVersion`)
- Added `target_value` column (stores: actual username, model ID, tag name, or version ID)
- Created index for fast queries by target
- Fully backwards compatible (old records with NULL targets preserved and functional)


## 1.6 New Feature - Image Limits <br />

**New Parameters:**
- `--max_images ` - Limit total images downloaded 
- `--max_per_model ` - Limit images per model in tag searches 

**Features:**
- Stops pagination early when limit reached (saves API calls)
- Only counts successful downloads (skipped images don't count toward limit)
- Clear progress reporting showing limit status
- Works with all modes (user, model, tag, model version)
- Fully backwards compatible (unlimited by default)
  

## 1.5 Bug Fixes <br />

1.  **Fixed Model Name Extraction (Bug #38):** <br />
    Images now correctly sort into model-specific folders instead of being incorrectly placed in `invalid_metadata`. The script now extracts model names from the CivitAI API's `civitaiResources` field when the legacy `Model` field is not present. <br />
    **Extraction Priority:** Existing Model field → civitaiResources checkpoint → baseModel → invalid_metadata <br />

2.  **Fixed Download Tracking Cross-Contamination (Bug #47):** <br />
    Database tracking keys now include query context (mode + target identifier) to prevent false "already downloaded" detections across different query types. Previously, an image downloaded via model search would be incorrectly skipped when downloading that same image via username search. <br />
    **New Key Format:** `{mode}:{target}_{image_id}_{quality}` (e.g., `username:Exorvious_12345_SD`) <br />
    Users can now collect complete image sets for each query type without cross-contamination. <br />
    
3.  **Fixed URN-Format Model Names (Bug #42):** <br />
    Images containing URN-format resource identifiers (e.g., `urn_air_sdxl_checkpoint_civitai_101055@128078`) in the Model field are now properly detected and replaced with human-readable model names extracted from `civitaiResources`. <br />

4.  **Enhanced Progress Visibility (Bug #50 - UX Improvement):** <br />
    Added status messages to clarify progress when downloading multiple identifiers concurrently. Previously, the progress bar appeared to "jump" between users/models, causing confusion about whether downloads were stuck. <br />
    Users now have clear visibility into which identifier is being processed, progress per page, and completion status. No more confusion about "stuck" downloads when multiple identifiers run concurrently <br />


## 1.4 Bug Fixes & New Feature <br />


1.  **Mode 4 Tag Filtering (New Feature):** <br />
    **Added `--filter_tags` argument:** Mode 4 (model version ID downloads) now supports filtering images by tags, similar to Mode 3. Users can specify one or more tags to only download images that match those tags. <br />
    **Prompt Check Support:** The `--disable_prompt_check` option now works with Mode 4 <br />
    
2.  **HTTP 429 Rate Limit Handling:** <br />
    Added proper handling for HTTP 429 (Too Many Requests) responses from the CivitAI API. The script now automatically retries with exponential backoff when rate limits are encountered, preventing failed downloads due to API throttling. <br />

4.  **Fixed Dynamic Retry Configuration:** <br />
    Fixed a bug where the `--retries` argument was not being properly respected due to static decorator evaluation. The retry count is now dynamically determined at runtime, allowing users to customize the number of retry attempts via command-line arguments. <br />

5.  **Enhanced Database Thread Safety:** <br />
    Enabled async locking in database read operations to prevent race conditions when multiple concurrent downloads check if an image has already been downloaded. This ensures data consistency in high-concurrency scenarios. <br />

6.  **Performance Optimization:** <br />
    Optimized debug logging to only execute expensive operations (metadata extraction, JSON parsing) when debug level logging is actually enabled, improving performance in production use. <br />


## 1.3 New Feature & Update <br />

1.  **Code Structure (Major Refactoring):**  <br />
                                           The entire script has been refactored into an object-oriented structure using the `CivitaiDownloader` class. This encapsulates state (configuration,tracking data, 
                                           statistics) and logic (downloading, sorting, API interaction) within the class, eliminating reliance on global variables. <br />

2.  **Scalable Tracking (SQLite Migration):** <br />
      **Replaced JSON:** The previous `downloaded_images.json` tracking file has been replaced with an **SQLite database** (`tracking_database.sqlite`). <br />
       **Relational Tags:** Image tags (for Mode 3 summaries) are now stored relationally in a separate `image_tags` table, linked to the main `tracked_images` table. This allows for efficient querying. <br />
       **Migration:** A separate `migrate_json_to_sqlite.py` script is provided for users to perform a one-time migration of their existing `downloaded_images.json` data into the new SQLite database format. <br />

3.  **Robust Error Handling & Retries:** <br />
       **Automatic Retries:** Integrated the `tenacity` library to automatically retry failed network operations (image downloads, API page fetches, model searches) caused by common transient issues like timeouts, 
                              connection errors, or specific server-side errors (500, 502, 503, 504). <br />
       **File Operations:** Implemented a `_safe_move` function with retries to handle potential file locking issues during sorting (especially on Windows). Added checks to verify move operations. <br />

4.  **Improved Tag Search (Mode 3) Validation:** <br />
       **Invalid Tag Detection:** When searching by tag, the script now fetches the first page of results and checks if any of the returned models *actually contain* the searched tag in their own metadata tags. <br />

5.  **Detailed Per-Identifier Statistics:** <br />
       **Granular Reporting:** The final statistics summary now provides a detailed breakdown for *each* identifier (username, model ID, tag, version ID) processed during the run. <br />

6.  **Improved User Interface & Experience:** <br />
       **Input Validation:** Added/improved validation loops for interactive inputs (e.g., ensuring numeric IDs, positive numbers). Handles invalid CLI arguments more gracefully (logs errors, exits). <br />
       **Clearer Output:** Refined console and log messages. Added specific warnings for invalid tags or identifiers that yield no results. Reduced console noise by logging successful per-file downloads only at the 
                            DEBUG level. Added a final summary note listing identifiers that resulted in no downloads. <br />



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
