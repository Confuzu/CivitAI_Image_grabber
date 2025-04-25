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
import sqlite3
from asyncio import Lock, Semaphore
import argparse
from typing import Optional, Tuple, List, Dict, Any, AsyncGenerator

# --- Tenacity for Retries ---
try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception, before_sleep_log, RetryError
except ImportError:
    print("Error: 'tenacity' library not found. Please install it using: pip install tenacity")
    sys.exit(1)

# --- anext compatibility ---
try:
    from asyncio import anext  # Python 3.10+
except ImportError:
    _MISSING = object()

    async def anext(ait: AsyncGenerator, default: Any = _MISSING) -> Any:
        try:
            return await ait.__anext__()
        except StopAsyncIteration:
            if default is _MISSING:
                raise
            return default

# ===================
# --- Constants ---
# ===================
BASE_API_URL: str = "https://civitai.com/api/v1/images"
MODELS_API_URL: str = "https://civitai.com/api/v1/models"
DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Content-Type": "application/json"
}
DEFAULT_SEMAPHORE_LIMIT: int = 5
DEFAULT_OUTPUT_DIR: str = "image_downloads"
DATABASE_FILENAME: str = "tracking_database.sqlite"  # <--- SQLite DB filename
LOG_FILENAME_TEMPLATE: str = "civit_image_downloader_log_{version}.txt"
SCRIPT_VERSION: str = "1.4-sqlite"
DEFAULT_TIMEOUT: int = 60
DEFAULT_RETRIES: int = 2
DEFAULT_MAX_PATH_LENGTH: int = 240
MAX_PAGINATION_PAGES: int = 100  # Prevent infinite loops in API pagination

# =======================
# --- Logging Setup ---
# =======================
# Configure logging globally for the script
script_dir: str = os.path.dirname(os.path.abspath(__file__))
log_file_path: str = os.path.join(script_dir, LOG_FILENAME_TEMPLATE.format(version=SCRIPT_VERSION))
logger: logging.Logger = logging.getLogger('CivitaiDownloader')
logger.setLevel(logging.INFO)
if not logger.handlers:
    # File Handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Formatter
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    console_handler.setFormatter(log_formatter)
    # Add Handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# =============================
# --- Retry Configuration ---
# =============================
# Configure the conditions under which tenacity should retry operations.
retry_logger: logging.Logger = logging.getLogger('CivitaiDownloader')
RETRYABLE_EXCEPTIONS: Tuple[type[Exception], ...] = (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, ConnectionResetError)
RETRYABLE_STATUS_CODES: set[int] = {500, 502, 503, 504}


def is_retryable_http_status(exception: BaseException) -> bool:
    return isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code in RETRYABLE_STATUS_CODES


def should_retry_exception(exception: BaseException) -> bool:
    return isinstance(exception, RETRYABLE_EXCEPTIONS) or is_retryable_http_status(exception)


# ===========================
# --- Argument Parser ---
# ===========================
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="CivitAI Image Downloader (SQLite Version)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP request timeout (seconds).")
    parser.add_argument("--quality", type=int, choices=[1, 2], help="Image quality: 1=SD, 2=HD.")
    parser.add_argument("--redownload", type=int, choices=[1, 2], default=2, help="Allow re-downloading tracked images: 1=Yes, 2=No.")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4], required=not sys.stdin.isatty(), help="Download mode (required if not interactive).")
    parser.add_argument("--tags", help="Tag(s) for Mode 3 (comma-separated).")
    parser.add_argument("--disable_prompt_check", choices=['y', 'n'], default='n', help="Disable prompt check in Mode 3 (y/n).")
    parser.add_argument("--username", help="Username(s) for Mode 1 (comma-separated).")
    parser.add_argument("--model_id", help="Model ID(s) for Mode 2 (comma-separated, numeric).")
    parser.add_argument("--model_version_id", help="Model Version ID(s) for Mode 4 (comma-separated, numeric).")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Base directory for downloads.")
    parser.add_argument("--semaphore_limit", type=int, default=DEFAULT_SEMAPHORE_LIMIT, help="Max concurrent downloads/API calls.")
    parser.add_argument("--no_sort", action='store_true', help="Disable sorting images into model subfolders.")
    parser.add_argument("--max_path", type=int, default=DEFAULT_MAX_PATH_LENGTH, help="Approximate max length for file paths.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Number of retries for failures.")
    return parser.parse_args()


# ==========================
# --- Helper Functions ---
# ==========================
def detect_extension(data: bytes) -> Optional[str]:
    """Detects file extension from magic numbers."""
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return ".png"
    if data.startswith(b'\xff\xd8\xff'):
        return ".jpeg"
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return ".webp"
    if len(data) >= 8 and data[4:8] == b'ftyp':
        return ".mp4"
    if data.startswith(b'\x1A\x45\xDF\xA3'):
        return ".webm"
    return None


# ============================
# --- The Downloader Class ---
# ============================
class CivitaiDownloader:
    """Handles downloading images and metadata from Civitai using SQLite tracking."""

    def __init__(self, args: argparse.Namespace):
        """Initializes the downloader with configuration and database connection."""
        self.args = args
        self.logger = logging.getLogger('CivitaiDownloader')
        self._interactive_mode_flag = self.args.mode is None

        # --- Configuration ---
        self.timeout: int = self._get_timeout_value()
        self.quality: str = self._get_quality()
        self.allow_redownload: int = self._get_redownload_option()
        self.mode: Optional[str] = self._get_mode_choice()
        self.output_dir: str = os.path.abspath(args.output_dir)
        self.semaphore_limit: int = self._get_semaphore_limit()
        self.disable_sorting: bool = args.no_sort
        self.max_path_length: int = args.max_path
        self.num_retries: int = args.retries

        # --- Database Setup ---
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.db_path: str = os.path.join(self.script_dir, DATABASE_FILENAME)
        self.db_conn: Optional[sqlite3.Connection] = None
        self._init_db()

        # --- API and Resources ---
        self.base_url: str = BASE_API_URL
        self.headers: Dict[str, str] = DEFAULT_HEADERS
        self.semaphore: Semaphore = Semaphore(self.semaphore_limit)
        self._client: Optional[httpx.AsyncClient] = None

        # --- State and Tracking ---
        self.tracking_lock: Lock = Lock()  # Lock for database write operations
        self.tag_model_mapping: Dict[str, List[Tuple[int, str]]] = {}
        self.tag_model_mapping_lock: Lock = Lock()
        self.visited_api_urls: set[str] = set()
        self.run_results: Dict[str, Dict[str, Any]] = {}
        self.skipped_reasons_summary: Dict[str, int] = {}
        self.failed_urls: List[str] = []
        self.failed_search_requests: List[str] = []

        # --- Log Initialized Config ---
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            self.logger.critical(f"Failed to create output directory '{self.output_dir}': {e}. Exiting.")
            print(f"CRITICAL ERROR: Cannot create output directory '{self.output_dir}'.")
            sys.exit(1)
        self.logger.info(f"--- Initializing Downloader ---")
        self.logger.info(f"Mode: {self.mode}, Quality: {self.quality}, Redownload: {'Yes' if self.allow_redownload == 1 else 'No'}")
        self.logger.info(f"Output Directory: {self.output_dir}")
        self.logger.info(f"Semaphore Limit: {self.semaphore_limit}, Timeout: {self.timeout}s, Retries Used: {self.num_retries}")
        self.logger.info(f"Sorting Disabled: {self.disable_sorting}, Max Path Length: {self.max_path_length}")
        self.logger.info(f"Tracking Database: {self.db_path}")
        self.logger.info(f"-------------------------------")

    # --- Database Initialization ---
    def _init_db(self) -> None:
        """Initializes the SQLite DB connection and creates tables if needed."""
        try:
            self.db_conn = sqlite3.connect(self.db_path, timeout=10)
            self.logger.info(f"Connected to tracking database: {self.db_path}")
            with self.db_conn:
                cursor = self.db_conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON;")  # Enable FK constraints
                # Create tracked_images table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tracked_images (
                        image_key TEXT PRIMARY KEY, 
                        image_id TEXT NOT NULL, 
                        quality TEXT NOT NULL,
                        path TEXT NOT NULL, 
                        download_date TEXT NOT NULL, 
                        url TEXT, 
                        checkpoint_name TEXT
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracked_images_key ON tracked_images (image_key)')
                # Create image_tags table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS image_tags (
                        image_key TEXT NOT NULL, 
                        tag TEXT NOT NULL,
                        PRIMARY KEY (image_key, tag),
                        FOREIGN KEY(image_key) REFERENCES tracked_images(image_key) ON DELETE CASCADE
                    ) WITHOUT ROWID;
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags (tag)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_tags_key ON image_tags (image_key)')
            self.logger.debug("Database schema initialized successfully (Relational Tags).")
        except sqlite3.Error as e:
            self.logger.critical(f"Database error during initialization: {e}", exc_info=True)
            print(f"CRITICAL ERROR: Failed to initialize tracking database '{self.db_path}'.")
            if self.db_conn:
                try:
                    self.db_conn.close()
                except sqlite3.Error:
                    pass
            sys.exit(1)

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily creates and returns the shared httpx.AsyncClient instance."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(headers=self.headers, timeout=self.timeout, follow_redirects=True)
            self.logger.debug("Created new httpx.AsyncClient")
        return self._client

    # --- Configuration Helper Methods ---
    def _is_interactive_mode(self) -> bool:
        return self._interactive_mode_flag

    def _get_timeout_value(self) -> int:
        """Gets the timeout value from args or interactive prompt."""
        cli_value_provided = self.args.timeout is not None and self.args.timeout != DEFAULT_TIMEOUT
        cli_value_is_valid = cli_value_provided and self.args.timeout > 0

        if cli_value_provided:
            if cli_value_is_valid:
                self.logger.warning(f"Timeout set via CLI argument: {self.args.timeout}")
                return self.args.timeout
            else:
                self.logger.warning(f"Invalid --timeout value '{self.args.timeout}' provided via CLI. Using default: {DEFAULT_TIMEOUT}s.")
                return DEFAULT_TIMEOUT
        elif self._is_interactive_mode():
            self.logger.debug("Interactive mode detected for timeout, prompting user...")
            while True:
                try:
                    timeout_input = input(f"Enter timeout value in seconds [default: {DEFAULT_TIMEOUT}]: ").strip()
                    if not timeout_input:
                        self.logger.warning(f"User selected default timeout: {DEFAULT_TIMEOUT}")
                        return DEFAULT_TIMEOUT
                    val = int(timeout_input)
                    if val > 0:
                        self.logger.debug(f"User selected timeout: {val}s")
                        return val
                    else:
                        print("Timeout must be a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a whole number.")
        else:
            self.logger.debug(f"Non-interactive mode, using default timeout: {DEFAULT_TIMEOUT}")
            return DEFAULT_TIMEOUT

    def _get_quality(self) -> str:
        if self.args.quality == 2:
            return 'HD'
        if self.args.quality == 1:
            return 'SD'
        elif self._is_interactive_mode():
            while True:
                inp = input("Choose quality (1=SD, 2=HD) [default: 1]: ").strip()
                if inp == '2':
                    return 'HD'
                if inp == '1' or inp == '':
                    return 'SD'
                print("Invalid choice.")
        else:
            return 'SD'

    def _get_redownload_option(self) -> int:
        if self.args.redownload == 1:
            return 1
        elif self._is_interactive_mode():
            self.logger.debug("Prompting user for redownload option...")
            while True:
                inp = input("Allow re-downloading tracked items? (1=Yes, 2=No) [default: 2]: ").strip()
                self.logger.debug(f"User input for redownload: '{inp}'")
                if inp == '1':
                    self.logger.debug("User selected redownload: Yes")
                    return 1
                if inp == '2' or inp == '':
                    self.logger.debug("User selected redownload: No")
                    return 2
                print("Invalid choice.")
        else:
            return self.args.redownload

    def _get_mode_choice(self) -> Optional[str]:
        if self.args.mode in [1, 2, 3, 4]:
            return str(self.args.mode)
        elif self._is_interactive_mode():
            while True:
                inp = input("Choose mode (1=user, 2=model ID, 3=tag search, 4=model version ID): ").strip()
                if inp in ['1', '2', '3', '4']:
                    return inp
                print("Invalid choice.")
        else:
            self.logger.error("Operation mode (--mode) is required in non-interactive mode.")
            print("Error: Operation mode (--mode) is required.")
            return None

    def _get_semaphore_limit(self) -> int:
        is_cli_non_default = (self.args.semaphore_limit is not None and
                              self.args.semaphore_limit != DEFAULT_SEMAPHORE_LIMIT and
                              self.args.semaphore_limit > 0)
        if is_cli_non_default:
            return self.args.semaphore_limit
        elif self._is_interactive_mode():
            self.logger.debug("Prompting user for semaphore limit...")
            while True:
                try:
                    inp = input(f"Enter max concurrent downloads [default: {DEFAULT_SEMAPHORE_LIMIT}]: ").strip()
                    if not inp:
                        self.logger.info(f"Using default semaphore: {DEFAULT_SEMAPHORE_LIMIT}")
                        return DEFAULT_SEMAPHORE_LIMIT
                    val = int(inp)
                    assert val > 0
                    self.logger.info(f"User set semaphore limit: {val}")
                    return val
                except (ValueError, AssertionError):
                    print("Invalid input. Must be positive number.")
        else:
            valid_val = self.args.semaphore_limit if self.args.semaphore_limit is not None and self.args.semaphore_limit > 0 else DEFAULT_SEMAPHORE_LIMIT
            self.logger.debug(f"Using default/parsed semaphore limit: {valid_val}")
            return valid_val

    # --- Result Dictionary Access Helper ---
    def _get_result_entry(self, parent_key: Optional[str], model_id: Optional[int] = None) -> Optional[Dict]:
        """Helper to get the correct dictionary entry in run_results to update stats."""
        if not parent_key:
            return None
        if parent_key not in self.run_results:
            self.logger.error(f"Attempted to get result entry for non-existent key: {parent_key}")
            return None
        if model_id is not None:
            model_key = f"model:{model_id}"
            if 'sub_details' not in self.run_results[parent_key]:
                self.run_results[parent_key]['sub_details'] = {}
            if model_key not in self.run_results[parent_key]['sub_details']:
                self.run_results[parent_key]['sub_details'][model_key] = {'success_count': 0, 'skipped_count': 0, 'no_meta_count': 0, 'api_items': 0, 'status': 'Pending', 'reason': None}
            return self.run_results[parent_key]['sub_details'][model_key]
        else:
            return self.run_results[parent_key]

    # --- Basic Client-Side Validation ---
    async def _validate_identifier_basic(self, identifier: str, id_type: str) -> Tuple[bool, Optional[str]]:
        """Performs basic client-side validation. Returns (is_valid, reason_if_invalid)."""
        if id_type in ['model', 'modelVersion']:
            if not identifier.isdigit() or int(identifier) <= 0:
                return False, f"Identifier '{identifier}' must be a positive number for type '{id_type}'."
        elif id_type == 'username':
            if not identifier or identifier.isspace():
                return False, "Username cannot be empty."
        elif id_type == 'tag':
            if not identifier or identifier.isspace():
                return False, "Tag cannot be empty."
        return True, None

    # --- SQLite Tracking Methods ---
    async def check_if_image_downloaded(self, image_id: str, quality: str) -> bool:
        """Checks if an image exists in the tracking database."""
        if not self.db_conn:
            return False
        image_key = f"{str(image_id)}_{quality}"
        query = "SELECT 1 FROM tracked_images WHERE image_key = ? LIMIT 1"
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(query, (image_key,))
            exists = cursor.fetchone() is not None
            return exists
        except sqlite3.Error as e:
            self.logger.error(f"DB error checking if image {image_key} downloaded: {e}", exc_info=True)
            return False

    async def mark_image_as_downloaded(self, image_id: str, image_path: str, quality: str, tags: Optional[List[str]] = None, url: Optional[str] = None, checkpoint_name: Optional[str] = None) -> None:
        """Marks image as downloaded in DB (INSERT OR REPLACE image, DELETE/INSERT tags)."""
        if not self.db_conn:
            return
        image_id_str = str(image_id)
        image_key = f"{image_id_str}_{quality}"
        current_date = datetime.now().strftime("%Y-%m-%d - %H:%M")
        tags = tags or []
        unique_tags = sorted(list(set(t for t in tags if t)))

        async with self.tracking_lock:
            cursor = self.db_conn.cursor()
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO tracked_images
                    (image_key, image_id, quality, path, download_date, url, checkpoint_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (image_key, image_id_str, quality, image_path, current_date, url, checkpoint_name))
                cursor.execute('DELETE FROM image_tags WHERE image_key = ?', (image_key,))
                if unique_tags:
                    tag_insert_data = [(image_key, tag) for tag in unique_tags]
                    cursor.executemany('INSERT INTO image_tags (image_key, tag) VALUES (?, ?)', tag_insert_data)
                self.db_conn.commit()
                self.logger.debug(f"Marked downloaded (in DB): {image_id_str} ({quality}) -> {image_path} (Tags: {len(unique_tags)})")
            except sqlite3.Error as e:
                self.logger.error(f"Database error marking image {image_key} as downloaded: {e}", exc_info=True)
                try:
                    self.db_conn.rollback()
                except sqlite3.Error as rb_e:
                    self.logger.error(f"Rollback failed: {rb_e}")

    # ===================================
    # --- Core Download/File Methods ---
    # ===================================
    async def download_image(self, image_api_item: Dict[str, Any], base_output_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Downloads a single image, detects extension, saves, with retries."""
        @retry(
            stop=stop_after_attempt(1 + self.num_retries),
            wait=wait_random_exponential(multiplier=1, max=10),
            retry=retry_if_exception(should_retry_exception),
            before_sleep=before_sleep_log(retry_logger, logging.WARNING)
        )
        async def _download_image_retryable():
            image_url = image_api_item.get('url')
            image_id = image_api_item.get('id')
            if not image_url or not image_id:
                return False, None, "Missing URL or ID in API data"
            if not base_output_path or not os.path.isdir(base_output_path):
                return False, None, f"Invalid target directory '{base_output_path}'"

            target_url = image_url
            if self.quality == 'HD':
                target_url = re.sub(r"width=\d{3,4}", "original=true", image_url)
                if target_url == image_url:
                    target_url += ('&' if '?' in target_url else '?') + "original=true"
                self.logger.debug(f"HD URL: {target_url}")

            final_image_path = None
            try:
                async with self.semaphore:
                    client = await self._get_client()
                    async with client.stream("GET", target_url) as response:
                        if 400 <= response.status_code < 500:
                            return False, None, f"HTTP Client Error {response.status_code} {response.reason_phrase}"
                        response.raise_for_status()

                        total_size = int(response.headers.get('content-length', 0))
                        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                        mapping = {"image/png": ".png", "image/jpeg": ".jpeg", "image/jpg": ".jpeg", "image/webp": ".webp", "video/mp4": ".mp4", "video/webm": ".webm"}
                        file_extension = mapping.get(content_type)
                        byte_iter = response.aiter_bytes()
                        first_chunk = await anext(byte_iter, None)
                        if first_chunk is None:
                            return False, None, "Downloaded empty file"
                        if file_extension is None:
                            file_extension = detect_extension(first_chunk)
                        if file_extension is None:
                            file_extension = ".png" if self.quality == 'HD' else ".jpeg"

                        filename_base = self._clean_path_component(str(image_id))
                        potential_final_path = os.path.join(base_output_path, filename_base + file_extension)
                        if len(potential_final_path) > self.max_path_length:
                            allowed_len = self.max_path_length - len(base_output_path) - 1
                            if allowed_len < 10:
                                return False, None, "Path too long, cannot shorten"
                            shortened_filename = self._clean_path_component(filename_base + file_extension, max_length=allowed_len)
                            final_image_path = os.path.join(base_output_path, shortened_filename)
                        else:
                            final_image_path = potential_final_path
                        final_dir = os.path.dirname(final_image_path)
                        if not final_dir:
                            return False, None, "Invalid final path structure"

                        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"DL {os.path.basename(final_image_path)}", leave=False, dynamic_ncols=True)
                        downloaded_size = 0
                        try:
                            os.makedirs(final_dir, exist_ok=True)
                            async with aiofiles.open(final_image_path, "wb") as file:
                                if first_chunk:
                                    await file.write(first_chunk)
                                    progress_bar.update(len(first_chunk))
                                    downloaded_size += len(first_chunk)
                                async for chunk in byte_iter:
                                    await file.write(chunk)
                                    progress_bar.update(len(chunk))
                                    downloaded_size += len(chunk)
                        finally:
                            progress_bar.close()

                        if total_size != 0 and downloaded_size < total_size:
                            try:
                                os.remove(final_image_path)
                            except OSError:
                                pass
                            return False, None, "Incomplete download"
                        self.logger.debug(f"Successfully downloaded: {final_image_path}")
                        return True, final_image_path, None
            except Exception as e:
                self.logger.error(f"Error downloading {target_url}: {e}", exc_info=True)
                raise

        try:
            return await _download_image_retryable()
        except RetryError as e:
            reason = f"Max retries exceeded: {e}"
            self.logger.error(f"Download failed for {image_api_item.get('url')} after retries: {e}")
            return False, None, reason

    async def _write_meta_data(self, meta: Optional[Dict[str, Any]], base_output_path_no_ext: str, image_id: str, username: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Writes metadata to a .txt file."""
        username = username or 'unknown_user'
        meta = meta or {}
        content_lines = []
        meta_filename_suffix = ""
        if not meta or all(str(v).strip() == '' for v in meta.values()):
            meta_filename_suffix = "_no_meta.txt"
            content_lines = ["No metadata available.", f"URL: https://civitai.com/images/{image_id}?username={username}"]
        else:
            meta_filename_suffix = "_meta.txt"
            content_lines = [f"{k}: {str(v) if v is not None else ''}" for k, v in meta.items()]

        directory = os.path.dirname(base_output_path_no_ext)
        base_filename = os.path.basename(base_output_path_no_ext)
        meta_filename = base_filename + meta_filename_suffix
        try:
            max_fname_len = self.max_path_length - len(directory) - 1
            if max_fname_len < 10:
                raise ValueError("Base directory path too long")
            cleaned_meta_filename = self._clean_path_component(meta_filename, max_length=max_fname_len)
            output_path_final = os.path.join(directory, cleaned_meta_filename)
        except Exception as path_e:
            self.logger.error(f"Error constructing metadata path for {base_filename}: {path_e}")
            return False, None

        try:
            if not directory:
                raise ValueError("Cannot determine directory")
            os.makedirs(directory, exist_ok=True)
            async with aiofiles.open(output_path_final, "w", encoding='utf-8') as f:
                await f.write("\n".join(content_lines))
                await f.flush()
            self.logger.debug(f"Wrote metadata: {output_path_final}")
            return True, output_path_final
        except Exception as e:
            self.logger.error(f"Error writing metadata to {output_path_final}: {e}", exc_info=True)
            return False, None

    # --- API Fetching Method ---
    async def _fetch_api_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetches a single page from the Civitai images API, with retries."""
        @retry(
            stop=stop_after_attempt(1 + self.num_retries),
            wait=wait_random_exponential(multiplier=1, max=10),
            retry=retry_if_exception(should_retry_exception),
            before_sleep=before_sleep_log(retry_logger, logging.WARNING)
        )
        async def _fetch_api_page_retryable():
            if url in self.visited_api_urls:
                return None
            self.visited_api_urls.add(url)
            client = await self._get_client()
            self.logger.debug(f"Fetching API page: {url}")
            try:
                async with self.semaphore:
                    response = await client.get(url)
                if 400 <= response.status_code < 500:
                    if response.status_code == 404:
                        self.logger.warning(f"API request returned 404 Not Found for {url}")
                        return None
                    else:
                        self.logger.error(f"API Client Error {response.status_code} for {url} (No Retry)")
                        self.failed_urls.append(url)
                        return None
                if response.status_code == 500:
                    try:
                        data = response.json()
                        error_msg = data.get('error', '').lower()
                        message_msg = data.get('message', '').lower()
                        if 'user not found' in error_msg:
                            self.logger.warning(f"API reported 'User not found' for {url} (Status 500)")
                            return data
                        if 'model not found' in error_msg or 'model not found' in message_msg:
                            self.logger.warning(f"API reported 'Model not found' for {url} (Status 500)")
                            return data
                        if 'version not found' in error_msg or 'version not found' in message_msg:
                            self.logger.warning(f"API reported 'Version not found' for {url} (Status 500)")
                            return data
                        response.raise_for_status()
                    except json.JSONDecodeError:
                        response.raise_for_status()
                response.raise_for_status()
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON Decode Error {url} (Status {response.status_code}): {e}")
                    self.failed_urls.append(url)
                    return None
            except Exception as e:
                self.logger.error(f"Error fetching API page {url}: {e}")
                raise

        try:
            return await _fetch_api_page_retryable()
        except RetryError as e:
            self.logger.error(f"API fetch failed after retries {url}: {e}")
            self.failed_urls.append(url)
            return None

    # --- Mode Execution Logic ---
    async def _process_api_items(self, items: List[Dict[str, Any]], target_dir: str, mode_specific_info: Optional[Dict] = None, parent_result_key: Optional[str] = None, model_id: Optional[int] = None) -> None:
        """Processes image items: checks skip conditions, creates download tasks."""
        tasks = []
        mode_specific_info = mode_specific_info or {}
        tag_to_check = mode_specific_info.get('tag_to_check')
        disable_prompt_check = mode_specific_info.get('disable_prompt_check', False)
        current_tag = mode_specific_info.get('current_tag')
        for item in items:
            image_id = item.get('id')
            if not image_id:
                continue
            should_skip, skip_reason = False, None
            result_entry = self._get_result_entry(parent_result_key, model_id)
            if self.allow_redownload == 2 and await self.check_if_image_downloaded(str(image_id), self.quality):
                skip_reason = "Already tracked"
                should_skip = True
            if not should_skip and tag_to_check and not disable_prompt_check:
                prompt = (item.get("meta") or {}).get("prompt", "").lower()
                if not all(word in prompt for word in tag_to_check.lower().split("_") if word):
                    skip_reason = f"Prompt check failed: {tag_to_check}"
                    should_skip = True
            if should_skip:
                if result_entry:
                    result_entry['skipped_count'] += 1
                if skip_reason:
                    self.skipped_reasons_summary[skip_reason] = self.skipped_reasons_summary.get(skip_reason, 0) + 1
                continue
            tasks.append(self._handle_single_download(item, target_dir, current_tag, parent_result_key, model_id))
        if tasks:
            await asyncio.gather(*tasks)

    async def _handle_single_download(self, item: Dict[str, Any], target_dir: str, current_tag: Optional[str] = None, parent_result_key: Optional[str] = None, model_id: Optional[int] = None) -> None:
        """Handles download, meta write, tracking, stats update for one item."""
        image_id = item.get('id')
        if not image_id:
            return
        base_path_no_ext = os.path.join(target_dir, self._clean_path_component(str(image_id)))
        result_entry = self._get_result_entry(parent_result_key, model_id)
        success, final_image_path, reason = await self.download_image(item, target_dir)
        if success and final_image_path:
            meta = item.get("meta", {})
            username = item.get("username")
            checkpoint_name = str(meta.get("Model", "")).strip() if meta and meta.get("Model") else None
            if result_entry:
                result_entry['success_count'] += 1
            tags_to_mark = [current_tag] if current_tag else []
            await self.mark_image_as_downloaded(str(image_id), final_image_path, self.quality, tags=tags_to_mark, url=item.get('url'), checkpoint_name=checkpoint_name)
            await self._write_meta_data(meta, base_path_no_ext, str(image_id), username)
            if not meta or all(str(v).strip() == '' for v in meta.values()):
                if result_entry:
                    result_entry['no_meta_count'] += 1
        elif not success:
            if result_entry:
                result_entry['skipped_count'] += 1
            fail_reason = reason or "Download failed"
            self.skipped_reasons_summary[fail_reason] = self.skipped_reasons_summary.get(fail_reason, 0) + 1

    async def _run_paginated_download(self, initial_url: str, target_dir: str, mode_specific_info: Optional[Dict] = None, parent_result_key: Optional[str] = None, model_id: Optional[int] = None) -> None:
        """Handles pagination and processing for an identifier, checking for 'Not Found' errors."""
        url: Optional[str] = initial_url
        page_count: int = 0
        identifier_status: str = 'Pending'
        identifier_reason: Optional[str] = None
        identifier_api_items: int = 0
        result_entry = self._get_result_entry(parent_result_key, model_id)
        if not result_entry:
            self.logger.error(f"Result entry missing for {parent_result_key}/{model_id}")
            return

        while url and page_count < MAX_PAGINATION_PAGES:
            page_count += 1
            self.logger.info(f"Requesting API page {page_count} for {os.path.basename(target_dir)}")
            page_data = await self._fetch_api_page(url)

            if page_count == 1 and page_data and isinstance(page_data, dict):
                error_msg = page_data.get('error', '').lower()
                message_msg = page_data.get('message', '').lower()
                not_found = False
                if 'user not found' in error_msg:
                    identifier_reason = "User not found (API Error)"
                    not_found = True
                elif 'model not found' in error_msg or 'model not found' in message_msg:
                    identifier_reason = "Model not found (API Error)"
                    not_found = True
                elif 'version not found' in error_msg or 'version not found' in message_msg:
                    identifier_reason = "Model Version not found (API Error)"
                    not_found = True

                if not_found:
                    self.logger.warning(f"Identifier not found for {parent_result_key}: {identifier_reason}")
                    identifier_status = 'Failed (Not Found)'
                    result_entry['status'] = identifier_status
                    result_entry['reason'] = identifier_reason
                    print(f"Error for '{parent_result_key}': {identifier_reason}. Please check the identifier.")
                    return

            if page_data is not None:
                items = page_data.get('items', [])
                metadata = page_data.get('metadata', {})
                item_count = len(items)
                identifier_api_items += item_count
                self.logger.info(f"Processing {item_count} items from page {page_count}")

                if items:
                    if identifier_status == 'Pending':
                        identifier_status = 'Processing'
                    await self._process_api_items(items, target_dir, mode_specific_info, parent_result_key, model_id)
                elif page_count == 1:
                    identifier_status = 'Completed (No Items Found)'
                    self.logger.warning(f"No items found for {os.path.basename(target_dir)} (User/Model/Version may have no images).")
                    print(f"Info for '{parent_result_key}': Identifier found, but no images associated with it.")

                url = metadata.get('nextPage')
                if not url:
                    if identifier_status != 'Failed (Not Found)':
                        if identifier_status == 'Processing':
                            identifier_status = 'Completed'
                        elif identifier_status == 'Pending':
                            identifier_status = 'Completed (No Items Found)'
                    self.logger.debug(f"No next page found for {os.path.basename(target_dir)}.")
                    break
                else:
                    await asyncio.sleep(1)
            else:
                fetch_fail_reason = f"Failed to fetch API page {page_count}"
                if url in self.failed_urls:
                    fetch_fail_reason += " (check logs for specific URL error)"
                if page_count == 1:
                    identifier_status = 'Failed'
                    identifier_reason = fetch_fail_reason
                else:
                    identifier_status = 'Completed (Fetch Error on Subsequent Page)'
                    identifier_reason = fetch_fail_reason
                self.logger.warning(f"Stopping pagination for {os.path.basename(target_dir)}: {fetch_fail_reason}.")
                break

        if result_entry and identifier_status != 'Failed (Not Found)':
            result_entry['status'] = identifier_status
            if identifier_reason and not result_entry.get('reason'):
                result_entry['reason'] = identifier_reason
            result_entry['api_items'] = identifier_api_items
            if model_id is not None and parent_result_key in self.run_results:
                self.run_results[parent_result_key]['api_items'] += identifier_api_items

        if not self.disable_sorting and identifier_status.startswith('Completed'):
            self.logger.info(f"Running sorting for: {target_dir}")
            await self._sort_images_by_model_name(target_dir)

    # --- Tag Search Methods ---
    @retry(
        stop=stop_after_attempt(1 + DEFAULT_RETRIES),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception(should_retry_exception),
        before_sleep=before_sleep_log(retry_logger, logging.WARNING)
    )
    async def _search_models_by_tag_page(self, url: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        """Helper: Fetches one page of model search results, with retries."""
        self.logger.debug(f"Fetching models page: {url}")
        async with self.semaphore:
            response = await client.get(url)
        if 400 <= response.status_code < 500:
            self.logger.error(f"Model search Client Error {response.status_code} for {url}")
            self.failed_search_requests.append(url)
            return None
        response.raise_for_status()
        try:
            return response.json()
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON Decode Error model search {url}: {e}")
            self.failed_search_requests.append(url)
            return None

    async def _search_models_by_tag(self, tag_query: str) -> List[Tuple[int, str]]:
        """
        Searches for model IDs and names by tag, handling pagination and adding
        validation based on first page results and a maximum page limit.
        """
        encoded_tag = tag_query.replace(" ", "%20")
        url: Optional[str] = f"{MODELS_API_URL}?tag={encoded_tag}&nsfw=True"
        models_found: List[Tuple[int, str]] = []
        model_ids_seen: set[int] = set()
        self.logger.info(f"Searching for models with tag: '{tag_query}' (Max pages: {MAX_PAGINATION_PAGES})")
        client = await self._get_client()
        visited_search_urls: set[str] = set()
        page_count: int = 0

        while url and page_count < MAX_PAGINATION_PAGES:
            page_count += 1
            if url in visited_search_urls:
                self.logger.warning(f"Model search loop detected: {url}")
                break
            visited_search_urls.add(url)
            try:
                data = await self._search_models_by_tag_page(url, client)
                if data:
                    items = data.get('items', [])
                    metadata = data.get('metadata', {})
                    if not items and page_count == 1:
                        self.logger.warning(f"No models found for tag '{tag_query}' on page 1.")
                        return []
                    elif items:
                        if page_count == 1:
                            tag_found_in_results = False
                            for model in items:
                                model_tags = {tag.lower() for tag in model.get('tags', []) if isinstance(tag, str)}
                                if tag_query.lower() in model_tags:
                                    tag_found_in_results = True
                                    break
                            if not tag_found_in_results:
                                msg = f"Tag '{tag_query}' not found in model tags on page 1. Aborting search (likely invalid tag)."
                                self.logger.warning(msg)
                                return []
                            else:
                                self.logger.debug(f"Tag '{tag_query}' validated via first page.")
                        new_models = 0
                        for model in items:
                            mid = model.get('id')
                            mname = model.get('name') or f"Unnamed {mid or '?'}"
                            if isinstance(mid, int) and mid not in model_ids_seen:
                                models_found.append((mid, mname))
                                model_ids_seen.add(mid)
                                new_models += 1
                        self.logger.debug(f"Found {new_models} new models for '{tag_query}' on page {page_count}.")
                    url = metadata.get('nextPage')
                    if url:
                        await asyncio.sleep(1)
                    else:
                        break
                else:
                    self.logger.warning(f"Stopping model search pagination for '{tag_query}' due to fetch failure page {page_count}.")
                    url = None
                    break
            except Exception as e:
                self.logger.error(f"Error processing model search page {url} for '{tag_query}': {e}")
                url = None
                break

        self.logger.info(f"Found {len(models_found)} unique models for tag '{tag_query}' after {page_count} pages.")
        return models_found

    # --- Main Execution Method ---
    async def run(self) -> None:
        """
        Main entry point to run the downloader based on initialized configuration.
        Collects identifiers, initializes results, dispatches tasks, and handles finalization.
        """
        self.logger.info(f"Starting Civitai Downloader run (Version {SCRIPT_VERSION})...")
        start_time = time.time()
        if not self.mode:
            self.logger.critical("No valid mode selected or initialized. Aborting run.")
            return

        self._check_mismatched_arguments()
        overall_success = True
        tasks = []
        identifiers_to_process: List[Tuple[str, str]] = []
        option_folder_map: Dict[str, str] = {
            'username': 'Username_Search',
            'model': 'Model_ID_Search',
            'modelVersion': 'Model_Version_ID_Search',
            'tag': 'Model_Tag_Search'
        }

        try:
            identifiers_raw: List[str] = []
            id_type: str = ""

            if self.mode == '1':
                id_type = 'username'
                identifiers_raw = self._get_usernames()
            elif self.mode == '2':
                id_type = 'model'
                identifiers_raw = self._get_model_ids()
            elif self.mode == '3':
                id_type = 'tag'
                identifiers_raw = self._get_tags()
            elif self.mode == '4':
                id_type = 'modelVersion'
                identifiers_raw = self._get_model_version_ids()
            else:
                raise ValueError(f"Internal Error: Invalid mode '{self.mode}' reached identifier collection.")

            if identifiers_raw and id_type:
                for ident_raw in identifiers_raw:
                    ident_clean = ident_raw.strip()
                    if ident_clean:
                        identifiers_to_process.append((id_type, ident_clean))
                if not identifiers_to_process:
                    raise ValueError(f"No valid identifiers found after processing input for mode {self.mode}.")
            else:
                raise ValueError(f"Failed to retrieve valid identifiers for mode {self.mode}.")

            for idt, ident in identifiers_to_process:
                result_key = f"{idt}:{ident}"
                self.run_results[result_key] = {
                    'success_count': 0, 'skipped_count': 0, 'no_meta_count': 0,
                    'api_items': 0, 'status': 'Pending', 'reason': None,
                    'sub_details': {}
                }

            option_folder = self._create_option_folder(option_folder_map[id_type])

            if self.mode == '3':
                disable_prompt_check = self._get_disable_prompt_check()
                option_folder = self._create_option_folder(option_folder_map['tag'])
                self.logger.info(f"--- Starting Tag Search Mode ---")
                tag_names_list = [ident for idt, ident in identifiers_to_process if idt == 'tag']
                self.logger.info(f"Tags: {tag_names_list}, Prompt Check: {'Disabled' if disable_prompt_check else 'Enabled'}")

                for id_type, tag in identifiers_to_process:
                    if id_type != 'tag':
                        continue
                    result_key = f"{id_type}:{tag}"
                    self.logger.info(f"--- Processing Tag: {tag} ---")
                    tag_query = tag.replace("_", " ")
                    sanitized_tag_dir_name = self._clean_path_component(tag)
                    tag_dir = os.path.join(option_folder, sanitized_tag_dir_name)
                    os.makedirs(tag_dir, exist_ok=True)
                    tag_tasks = []

                    try:
                        found_models = await self._search_models_by_tag(tag_query)
                        if not found_models:
                            self.logger.warning(f"No models found or processed for tag '{tag}'.")
                            if self.run_results[result_key]['status'] == 'Pending':
                                self.run_results[result_key]['status'] = 'Completed (No Models Found or Invalid Tag)'
                            continue

                        async with self.tag_model_mapping_lock:
                            self.tag_model_mapping.setdefault(tag, []).extend(found_models)

                        self.run_results[result_key]['status'] = 'Processing Models'
                        self.logger.info(f"Found {len(found_models)} models for '{tag}'. Queueing downloads...")
                        tag_to_check = tag if not disable_prompt_check else None

                        for model_id, model_name in found_models:
                            self.logger.debug(f"Queueing task for model ID: {model_id} ('{model_name}') under tag '{tag}'")
                            model_target_dir = os.path.join(tag_dir, f"model_{model_id}")
                            os.makedirs(model_target_dir, exist_ok=True)
                            url = f"{self.base_url}?modelId={model_id}&nsfw=X"
                            mode_info = {'tag_to_check': tag_to_check, 'disable_prompt_check': disable_prompt_check, 'current_tag': tag}
                            tag_tasks.append(self._run_paginated_download(url, model_target_dir, mode_info, parent_result_key=result_key, model_id=model_id))

                        if tag_tasks:
                            self.logger.info(f"Executing asyncio.gather for {len(tag_tasks)} tasks (tag '{tag}')...")
                            results = await asyncio.gather(*tag_tasks, return_exceptions=True)
                            for i, result in enumerate(results):
                                if isinstance(result, Exception):
                                    self.logger.error(f"Task {i} for tag '{tag}' failed within gather: {result}", exc_info=result)
                                    overall_success = False
                        else:
                            self.logger.warning(f"No tasks generated for tag '{tag}', skipping gather.")

                        if self.run_results[result_key]['status'] == 'Processing Models':
                            self.run_results[result_key]['status'] = 'Completed'

                    except Exception as tag_proc_err:
                        self.logger.error(f"Error processing tag {tag}: {tag_proc_err}", exc_info=True)
                        if result_key in self.run_results:
                            self.run_results[result_key]['status'] = 'Failed'
                            self.run_results[result_key]['reason'] = f"{type(tag_proc_err).__name__}: {tag_proc_err}"
                        overall_success = False

                self.logger.info(f"--- Finished Tag Search Mode ---")
            else:
                for idt, ident in identifiers_to_process:
                    result_key = f"{idt}:{ident}"
                    valid, reason = await self._validate_identifier_basic(ident, idt)
                    if not valid:
                        self.run_results[result_key].update({'status': 'Failed (Validation)', 'reason': reason})
                        continue
                    target_dir, url = "", ""
                    url_params = f"&nsfw=X&sort=Newest" if idt == 'username' else "&nsfw=X"
                    if idt == 'username':
                        target_dir = os.path.join(option_folder, self._clean_path_component(ident))
                        url = f"{self.base_url}?username={ident}{url_params}"
                    elif idt == 'model':
                        target_dir = os.path.join(option_folder, f"model_{ident}")
                        url = f"{self.base_url}?modelId={ident}{url_params}"
                    elif idt == 'modelVersion':
                        target_dir = os.path.join(option_folder, f"modelVersion_{ident}")
                        url = f"{self.base_url}?modelVersionId={ident}{url_params}"
                    if target_dir and url:
                        os.makedirs(target_dir, exist_ok=True)
                        tasks.append(self._run_paginated_download(url, target_dir, parent_result_key=result_key))
                    else:
                        self.run_results[result_key].update({'status': 'Failed', 'reason': 'Internal setup error'})
                if tasks:
                    self.logger.info(f"Executing {len(tasks)} download tasks (Modes 1, 2, 4)...")
                    await asyncio.gather(*tasks)
                    self.logger.info("Finished gathering non-tag download tasks.")
                else:
                    self.logger.info("No download tasks were generated for modes 1, 2, or 4.")

        except ValueError as ve:
            self.logger.critical(f"Run setup error: {ve}", exc_info=False)
            print(f"\nSETUP ERROR: {ve}")
            overall_success = False
        except Exception as e:
            self.logger.critical(f"Critical error during run setup/dispatch: {e}", exc_info=True)
            print(f"\n--- CRITICAL ERROR ---")
            print(f"An unexpected error stopped the process: {e}")
            print(f"Please check the log file for details: {log_file_path}")
            print(f"----------------------")
            overall_success = False
            for key, data in self.run_results.items():
                if data.get('status') == 'Pending':
                    data.update({'status': 'Failed', 'reason': f'Run interrupted by critical error: {e}'})
        finally:
            self.logger.info("Run finalization steps...")
            final_option_folder = ""
            if self.mode and self.mode in option_folder_map:
                final_option_folder = os.path.join(self.output_dir, option_folder_map[self.mode])

            if self._client and not self._client.is_closed:
                await self._client.aclose()
                self.logger.info("HTTP Client closed.")
            if self.db_conn:
                try:
                    self.db_conn.close()
                    self.logger.info("Database connection closed.")
                except sqlite3.Error as e:
                    self.logger.error(f"Error closing database connection: {e}")

            if self.mode == '3' and final_option_folder:
                await self._write_tag_summaries(final_option_folder)

            self._print_download_statistics()

            failed_items = {k: v for k, v in self.run_results.items() if not str(v.get('status', '')).startswith('Completed')}
            if failed_items:
                self.logger.warning("Some identifiers failed processing or completed with errors:")
                print("\nWarning: Some identifiers had issues:")
                for key in sorted(failed_items.keys()):
                    data = failed_items[key]
                    reason = data.get('reason', 'N/A')
                    self.logger.warning(f"- {key}: Status={data['status']}, Reason={reason}")
                    print(f"- {key}: {data['status']} (Reason: {reason})")

            if self.failed_urls:
                unique_failed_urls = set(self.failed_urls)
                msg = f"Failed to fetch {len(unique_failed_urls)} unique API page URLs after retries."
                self.logger.warning(msg + " Check DEBUG logs for specific URLs.")
                print(f"\nWarning: {msg} (see log for details).")
            if self.failed_search_requests:
                unique_failed_searches = set(self.failed_search_requests)
                msg = f"Failed to fetch {len(unique_failed_searches)} unique model search URLs after retries."
                self.logger.warning(msg + " Check DEBUG logs for specific URLs.")
                print(f"Warning: {msg} (see log for details).")

            run_duration = time.time() - start_time
            self.logger.info(f"Total run duration: {run_duration:.2f} seconds")
            run_status_msg = 'successfully' if overall_success and not failed_items else 'with errors'
            self.logger.info(f"Run finished {run_status_msg}.")
            print("\nDownload process complete.")

    # --- Input Gathering Methods ---
    def _get_usernames(self) -> List[str]:
        """Gets usernames from args or interactive prompt."""
        if self.args.username:
            return [n.strip() for n in self.args.username.split(",") if n.strip()]
        elif self._is_interactive_mode():
            while True:
                names = input("Enter username(s) (comma-separated): ").strip()
                if names:
                    return [n.strip() for n in names.split(",") if n.strip()]
                else:
                    print("Please enter at least one username.")
        else:
            self.logger.error("Username (--username) required for Mode 1 in non-interactive mode.")
            print("Error: Username required.")
            sys.exit(1)

    def _get_model_ids(self) -> List[str]:
        """Gets model IDs from args or interactive prompt, validating numeric."""
        if self.args.model_id:
            ids_str = str(self.args.model_id)
            ids = [i.strip() for i in ids_str.split(',') if i.strip()]
            if ids and all(i.isdigit() for i in ids):
                return ids
            self.logger.error(f"Invalid Model ID provided via --model_id: '{self.args.model_id}'. Must be numeric, comma-separated.")
            print(f"Error: Invalid Model ID provided via --model_id: '{self.args.model_id}'. Must be numeric.")
            sys.exit(1)
        elif self._is_interactive_mode():
            while True:
                ids_in = input("Enter model ID(s) (numeric, comma-separated): ").strip()
                ids = [i.strip() for i in ids_in.split(',') if i.strip()]
                if not ids:
                    print("Please enter at least one model ID.")
                    continue
                if all(i.isdigit() for i in ids):
                    return ids
                else:
                    print("Invalid input. Please enter numeric IDs only, separated by commas.")
        else:
            self.logger.error("Model ID (--model_id) required for Mode 2 in non-interactive mode.")
            print("Error: Model ID required for Mode 2.")
            sys.exit(1)

    def _get_tags(self) -> List[str]:
        """Gets tags from args or interactive prompt, replacing spaces with underscores."""
        tags_raw = None
        if self.args.tags:
            tags_raw = self.args.tags
        elif self._is_interactive_mode():
            while True:
                tags_in = input("Enter tags (comma-separated): ").strip()
                if tags_in:
                    tags_raw = tags_in
                    break
                else:
                    print("Please enter at least one tag.")
        else:
            self.logger.error("Tags (--tags) required for Mode 3 in non-interactive mode.")
            print("Error: Tags required.")
            sys.exit(1)
        tags = [t.strip().replace(" ", "_") for t in tags_raw.split(',') if t.strip()]
        if not tags:
            self.logger.error("No valid tags found after processing input.")
            print("Error: No valid tags provided.")
            sys.exit(1)
        return tags

    def _get_disable_prompt_check(self) -> bool:
        """Gets the disable_prompt_check option (True/False) from args or interactive prompt."""
        val_map = {'y': True, 'n': False}
        argparse_default = 'n'

        cli_value_provided = self.args.disable_prompt_check is not None
        cli_value_is_non_default = cli_value_provided and self.args.disable_prompt_check.lower() != argparse_default

        if cli_value_provided and cli_value_is_non_default:
            self.logger.debug(f"Disable prompt check explicitly set via CLI: {self.args.disable_prompt_check}")
            return val_map.get(self.args.disable_prompt_check.lower(), False)
        elif self._is_interactive_mode():
            self.logger.debug("Interactive mode detected for disable_prompt_check, prompting user...")
            while True:
                resp = input("Disable prompt check? (y/n) [default: n]: ").lower().strip()
                self.logger.debug(f"User input for disable_prompt_check: '{resp}'")
                if resp == 'y':
                    self.logger.info("User selected disable prompt check: Yes (y)")
                    return True
                elif resp == 'n' or resp == '':
                    self.logger.info(f"User selected disable prompt check: No (n) (Selected: '{resp}')")
                    return False
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
        else:
            final_value = val_map.get(self.args.disable_prompt_check.lower(), val_map[argparse_default])
            self.logger.debug(f"Non-interactive mode, using default/parsed disable_prompt_check value: {final_value}")
            return final_value

    def _get_model_version_ids(self) -> List[str]:
        """Gets model version IDs from args or interactive prompt, validating numeric."""
        if self.args.model_version_id:
            ids_str = str(self.args.model_version_id)
            ids = [i.strip() for i in ids_str.split(',') if i.strip()]
            if ids and all(i.isdigit() for i in ids):
                return ids
            self.logger.error(f"Invalid Model Version ID provided via --model_version_id: '{self.args.model_version_id}'. Must be numeric.")
            print(f"Error: Invalid Model Version ID provided via --model_version_id: '{self.args.model_version_id}'.")
            sys.exit(1)
        elif self._is_interactive_mode():
            while True:
                ids_in = input("Enter model version ID(s) (numeric, comma-separated): ").strip()
                ids = [i.strip() for i in ids_in.split(',') if i.strip()]
                if not ids:
                    print("Please enter at least one model version ID.")
                    continue
                if all(i.isdigit() for i in ids):
                    return ids
                else:
                    print("Invalid input. Please enter numeric IDs only, separated by commas.")
        else:
            self.logger.error("Model Version ID (--model_version_id) required for Mode 4 in non-interactive mode.")
            print("Error: Model Version ID required for Mode 4.")
            sys.exit(1)

    # --- Utility and Reporting Methods ---
    def _create_option_folder(self, option_name: str) -> str:
        """Creates mode-specific subfolder within output directory."""
        option_dir = os.path.join(self.output_dir, option_name)
        try:
            os.makedirs(option_dir, exist_ok=True)
            self.logger.debug(f"Ensured folder: {option_dir}")
            return option_dir
        except OSError as e:
            self.logger.error(f"Failed create folder '{option_dir}': {e}")
            return self.output_dir

    def _clean_path_component(self, path_part: str, max_length: Optional[int] = None) -> str:
        """Cleans/shortens a filename or directory name component using simple replacement."""
        if max_length is None:
            max_length = self.max_path_length

        invalid_char_set = set('<>:"/\\|?*\t\n\r') | {chr(i) for i in range(32)}
        cleaned = path_part.replace("%20", " ").replace("%2B", "+").replace("%26", "&")

        cleaned_list = []
        for char in cleaned:
            if char in invalid_char_set:
                cleaned_list.append('_')
            else:
                cleaned_list.append(char)
        cleaned = "".join(cleaned_list)

        cleaned = cleaned.strip('. _')
        cleaned = re.sub(r'_+', '_', cleaned)

        original_len = len(cleaned)
        if original_len > max_length:
            name, ext = os.path.splitext(cleaned)
            max_name_length = max(1, max_length - len(ext))
            truncated_name = name[:max_name_length].strip('_')
            cleaned = truncated_name + ext
            if len(cleaned) > max_length:
                cleaned = cleaned[:max_length].strip('_')

        final_cleaned = cleaned if cleaned else "_"
        return final_cleaned

    def _safe_move(self, src: str, dst: str, max_retries: int = 5, delay: float = 0.5) -> bool:
        """Robust file moving with retries and logging, checking actual outcome."""
        abs_src = os.path.abspath(src)
        abs_dst = os.path.abspath(dst)
        src_basename = os.path.basename(abs_src)

        self.logger.debug(f"Safe move requested for '{src_basename}':")
        self.logger.debug(f"  Source Abs: {abs_src}")
        self.logger.debug(f"  Dest Abs:   {abs_dst}")

        dst_dir = os.path.dirname(abs_dst)
        try:
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
                self.logger.debug(f"Created destination directory: {dst_dir}")
        except OSError as e:
            self.logger.error(f"Failed to create destination directory '{dst_dir}' for move: {e}")
            return False

        if not os.path.exists(abs_src):
            self.logger.warning(f"Source file does not exist at start of safe_move: {abs_src}")
            return False

        move_succeeded_flag = False
        for attempt in range(1, max_retries + 1):
            try:
                if not os.path.exists(abs_src):
                    self.logger.warning(f"Source file vanished before move attempt {attempt}: {abs_src}")
                    return False

                self.logger.debug(f"Attempting shutil.move (Attempt {attempt}/{max_retries})...")
                shutil.move(abs_src, abs_dst)

                if os.path.exists(abs_dst):
                    if not os.path.exists(abs_src):
                        move_succeeded_flag = True
                        self.logger.debug(f"Move successful (dest exists, src gone): '{src_basename}' -> '{os.path.relpath(abs_dst, self.output_dir)}'")
                        return True
                    else:
                        self.logger.warning(f"shutil.move completed for '{src_basename}', BUT source file still exists at '{abs_src}'. Destination is '{abs_dst}'. Treating as copy, not move.")
                        try:
                            self.logger.warning(f"Attempting to remove source '{abs_src}' after copy-like behavior.")
                            os.remove(abs_src)
                            if not os.path.exists(abs_src):
                                self.logger.info(f"Successfully removed source '{abs_src}' after copy-like move.")
                                move_succeeded_flag = True
                                return True
                            else:
                                self.logger.error(f"Failed to remove source '{abs_src}' after copy-like move.")
                                return False
                        except OSError as rm_err:
                            self.logger.error(f"Error removing source '{abs_src}' after copy-like move: {rm_err}")
                            return False
                else:
                    self.logger.error(f"shutil.move completed for '{src_basename}' without error, BUT destination file '{abs_dst}' does NOT exist. Source exists: {os.path.exists(abs_src)}")
                    move_succeeded_flag = False
                    return False
            except (PermissionError, OSError) as e:
                if attempt < max_retries:
                    self.logger.debug(f"Move attempt {attempt}/{max_retries} failed for '{src_basename}' ({e}), retrying in {delay * attempt:.1f}s...")
                    time.sleep(delay * attempt)
                else:
                    self.logger.error(f"Failed to move '{src_basename}' after {max_retries} attempts: {e}")
                    return False
            except Exception as e_unexp:
                self.logger.error(f"Unexpected error during safe_move for '{src_basename}' attempt {attempt}: {e_unexp}", exc_info=True)
                if attempt < max_retries:
                    time.sleep(delay * attempt)
                else:
                    return False

        self.logger.error(f"safe_move loop finished unexpectedly for '{src_basename}'. Final status uncertain.")
        return move_succeeded_flag

    def _print_download_statistics(self) -> None:
        """Prints the final download statistics summary."""
        print("\n--- Download Statistics Summary ---")
        if not self.run_results:
            print("No identifiers processed in this run.")
            print("---------------------------------\n")
            return

        identifier_final_counts = {}
        no_models_processed_identifiers = []
        no_process_statuses = {
            'Completed (No Models Found)',
            'Completed (No Models Found or Invalid Tag)',
            'Completed (No Items Found)',
            'Failed (Validation)',
            'Failed (Not Found)'
        }

        for key, data in self.run_results.items():
            agg_api = data.get('api_items', 0)
            agg_dl = data.get('success_count', 0)
            agg_skip = data.get('skipped_count', 0)
            agg_nometa = data.get('no_meta_count', 0)

            if key.startswith('tag:') and 'sub_details' in data:
                agg_api = sum(sd.get('api_items', 0) for sd in data['sub_details'].values())
                agg_dl = sum(sd.get('success_count', 0) for sd in data['sub_details'].values())
                agg_skip = sum(sd.get('skipped_count', 0) for sd in data['sub_details'].values())
                agg_nometa = sum(sd.get('no_meta_count', 0) for sd in data['sub_details'].values())

            identifier_final_counts[key] = {
                'api': agg_api, 'dl': agg_dl, 'skip': agg_skip, 'nometa': agg_nometa
            }

            current_status = data.get('status', 'Unknown')
            is_no_process_status = current_status in no_process_statuses
            all_counts_zero = (agg_api == 0 and agg_dl == 0 and agg_skip == 0 and agg_nometa == 0)

            if is_no_process_status and all_counts_zero:
                no_models_processed_identifiers.append(key)

        total_api_items = sum(counts['api'] for counts in identifier_final_counts.values())
        total_downloaded = sum(counts['dl'] for counts in identifier_final_counts.values())
        total_skipped = sum(counts['skip'] for counts in identifier_final_counts.values())
        total_no_meta = sum(counts['nometa'] for counts in identifier_final_counts.values())

        print(f"Total API items processed (approx): {total_api_items}")
        print(f"Total successful downloads this run: {total_downloaded}")
        print(f"Total images without metadata: {total_no_meta}")
        print(f"Total skipped/failed items: {total_skipped}")

        if self.skipped_reasons_summary:
            print("\nReasons for skipping/failing items across run:")
            sorted_reasons = sorted(self.skipped_reasons_summary.items(), key=lambda item: item[1], reverse=True)
            for reason, count in sorted_reasons:
                print(f"- {reason}: {count} times")

        print("\n--- Results per Identifier ---")
        sorted_keys = sorted(self.run_results.keys())
        for key in sorted_keys:
            data = self.run_results[key]
            counts = identifier_final_counts[key]
            try:
                id_type, identifier = key.split(":", 1)
            except ValueError:
                identifier, id_type = key, "Unknown"

            print(f"Identifier: {identifier} (Type: {id_type})")
            print(f"  Status: {data.get('status', 'Unknown')}")
            if data.get('reason'):
                print(f"  Reason: {data['reason']}")
            print(f"  API Items: {counts['api']}")
            print(f"  Downloaded: {counts['dl']}")
            print(f"  Skipped/Failed: {counts['skip']}")
            print(f"  No Metadata: {counts['nometa']}")
            print("-" * 10)

        if no_models_processed_identifiers:
            print("\nNOTE: The following identifiers resulted in zero models/images being processed:")
            for key in sorted(no_models_processed_identifiers):
                status = self.run_results[key].get('status', 'Unknown')
                reason = self.run_results[key].get('reason')
                reason_str = f" (Reason: {reason})" if reason else ""
                print(f"- {key} (Status: {status}{reason_str})")

        print("---------------------------------\n")
        self.logger.info(f"Run Stats Aggregated: Success={total_downloaded}, Skipped={total_skipped}, NoMeta={total_no_meta}, API Items={total_api_items}")

    def _check_mismatched_arguments(self) -> None:
        """Logs warnings if CLI arguments conflict with the selected mode."""
        if not self.mode or self.mode not in ['1', '2', '3', '4']:
            return
        mode = int(self.mode)
        relevant_args = []
        unused_args = []
        if mode == 1:
            relevant_args = ['username']
        elif mode == 2:
            relevant_args = ['model_id']
        elif mode == 3:
            relevant_args = ['tags', 'disable_prompt_check']
        elif mode == 4:
            relevant_args = ['model_version_id']
        all_mode_args = ['username', 'model_id', 'model_version_id', 'tags', 'disable_prompt_check']
        for argn in all_mode_args:
            argval = getattr(self.args, argn, None)
            is_default = False
            if argn == 'disable_prompt_check':
                is_default = (argval == 'n')
            if argval is not None and not is_default and argn not in relevant_args:
                unused_args.append(f"--{argn.replace('_', '-')}")
        if unused_args:
            msg = f"Warning: Arguments potentially unused in mode {mode}: {', '.join(unused_args)}"
            self.logger.warning(msg)
            print(msg)

    # ============================
    # --- Sorting Logic ---
    # ============================
    async def _sort_images_by_model_name(self, base_dir: str) -> None:
        """Sorts downloaded images and metadata files into subfolders based on metadata."""
        self.logger.info(f"Starting sort process in: {base_dir}")
        if not os.path.isdir(base_dir):
            self.logger.warning(f"Sort directory does not exist: {base_dir}")
            return

        no_meta_dir = os.path.join(base_dir, 'no_metadata')
        invalid_meta_dir = os.path.join(base_dir, 'invalid_metadata')
        processed_filenames: set[str] = set()

        try:
            all_entries = list(os.scandir(base_dir))
            all_files_in_basedir = {entry.name for entry in all_entries if entry.is_file()}
        except OSError as e:
            self.logger.error(f"Cannot list files in sort directory {base_dir}: {e}")
            return

        meta_files = [f for f in all_files_in_basedir if f.endswith(('_meta.txt', '_no_meta.txt'))]

        if not meta_files:
            self.logger.info(f"No metadata files found to process in {base_dir}")
            return

        os.makedirs(no_meta_dir, exist_ok=True)
        os.makedirs(invalid_meta_dir, exist_ok=True)
        self.logger.debug(f"Found {len(meta_files)} meta files to process in {base_dir}")

        for meta_file in meta_files:
            if meta_file in processed_filenames:
                self.logger.debug(f"Skipping already processed file: {meta_file}")
                continue

            meta_path = os.path.join(base_dir, meta_file)
            base_name: Optional[str] = None

            if meta_file.endswith('_no_meta.txt'):
                base_name = meta_file.rsplit('_no_meta.txt', 1)[0]
            elif meta_file.endswith('_meta.txt'):
                base_name = meta_file.rsplit('_meta.txt', 1)[0]

            if not base_name:
                self.logger.warning(f"Could not extract base name from meta file: {meta_file}. Skipping.")
                continue

            target_dir_for_pair: Optional[str] = None
            self.logger.debug(f"Processing pair based on meta file: {meta_file} (base name: {base_name})")

            try:
                if meta_file.endswith('_no_meta.txt'):
                    target_dir_for_pair = no_meta_dir
                    self.logger.debug(f"Target for '{meta_file}' is no_metadata folder.")
                elif meta_file.endswith('_meta.txt'):
                    model_name = None
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        match = re.search(r"^Model:\s*(.+?)\s*$", content, re.IGNORECASE | re.MULTILINE)
                        if match:
                            model_name = match.group(1).strip()
                        self.logger.debug(f"Read '{meta_file}'. Found model: {model_name}")
                    except FileNotFoundError:
                        self.logger.warning(f"Meta file vanished before reading: {meta_path}. Skipping.")
                        continue
                    except Exception as read_e:
                        self.logger.error(f"Failed to read metadata file {meta_path}: {read_e}")
                        model_name = None

                    if model_name and model_name != 'unknown':
                        model_subdir_name = self._clean_path_component(model_name, max_length=60)
                        target_dir_for_pair = os.path.join(base_dir, model_subdir_name)
                        os.makedirs(target_dir_for_pair, exist_ok=True)
                        self.logger.debug(f"Target for '{meta_file}' is model folder: '{model_subdir_name}'")
                    else:
                        target_dir_for_pair = invalid_meta_dir
                        self.logger.debug(f"Target for '{meta_file}' is invalid_metadata folder.")

                if not target_dir_for_pair:
                    self.logger.error(f"Logic error: Could not determine target directory for {meta_file}. Skipping.")
                    continue

                target_meta_path = os.path.join(target_dir_for_pair, meta_file)
                self.logger.debug(f"Attempting move: {meta_path} -> {target_meta_path}")
                if self._safe_move(meta_path, target_meta_path):
                    processed_filenames.add(meta_file)
                    self.logger.debug(f"Meta move successful for {meta_file}")

                    possible_image_found = False
                    image_moved = False
                    image_extensions = ['.jpeg', '.png', '.webp', '.mp4', '.webm']
                    for ext in image_extensions:
                        image_file = base_name + ext
                        if image_file in all_files_in_basedir and image_file not in processed_filenames:
                            possible_image_found = True
                            image_path = os.path.join(base_dir, image_file)
                            target_image_path = os.path.join(target_dir_for_pair, image_file)
                            self.logger.debug(f"Found potential image '{image_file}'. Attempting move to: {target_dir_for_pair}")
                            if self._safe_move(image_path, target_image_path):
                                processed_filenames.add(image_file)
                                image_moved = True
                                self.logger.debug(f"Image move successful for {image_file}")
                            else:
                                self.logger.warning(f"Failed to move image file {image_file} for meta {meta_file}")
                            break
                    if not possible_image_found:
                        self.logger.debug(f"No corresponding image found in {base_dir} for base name '{base_name}' (checked: {image_extensions})")
                    elif not image_moved:
                        self.logger.warning(f"Image file found for {meta_file} but failed to move.")
                else:
                    self.logger.warning(f"Failed to move meta file {meta_file}, skipping associated image move.")
            except Exception as e:
                self.logger.error(f"Unexpected error during sort processing for {meta_file}: {e}", exc_info=True)

        self.logger.debug("Checking for orphaned files after sorting...")
        try:
            current_files_after_sort = {f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))}
            remaining_files = current_files_after_sort - processed_filenames - {f for f in current_files_after_sort if f.lower().endswith('.csv')}

            orphaned_images = []
            image_extensions_set = {'.jpeg', '.png', '.webp', '.mp4', '.webm'}
            for fname in remaining_files:
                base_name_orphan, ext_orphan = os.path.splitext(fname)
                if ext_orphan.lower() in image_extensions_set and base_name_orphan.isdigit():
                    meta_exists = False
                    for meta_suffix in ['_meta.txt', '_no_meta.txt']:
                        possible_meta = base_name_orphan + meta_suffix
                        locations_to_check = [base_dir, no_meta_dir, invalid_meta_dir]
                        potential_model_dirs = [d.path for d in os.scandir(base_dir) if d.is_dir() and d.name not in ['no_metadata', 'invalid_metadata']]
                        locations_to_check.extend(potential_model_dirs)
                        for loc in locations_to_check:
                            if os.path.exists(os.path.join(loc, possible_meta)):
                                meta_exists = True
                                break
                        if meta_exists:
                            break
                    if not meta_exists:
                        orphaned_images.append(fname)
            if orphaned_images:
                self.logger.warning(f"Found potential orphaned image files in {base_dir}: {orphaned_images}")
        except OSError as e:
            self.logger.warning(f"Could not perform orphan check in {base_dir}: {e}")

        self.logger.info(f"Finished sort process in: {base_dir}")

    # ============================
    # --- Tag Summary CSV ---
    # ============================
    async def _write_tag_summaries(self, option_folder: str) -> None:
        """Writes CSV summaries for tags using SQLite data."""
        self.logger.info("Generating tag summary CSV files...")
        if not self.db_conn:
            self.logger.error("DB connection unavailable.")
            return
        async with self.tag_model_mapping_lock:
            tags_processed = list(self.tag_model_mapping.keys())
        if not tags_processed:
            self.logger.info("No tags processed, skipping CSV.")
            return
        for tag in tags_processed:
            sanitized_tag = self._clean_path_component(tag)
            tag_dir = os.path.join(option_folder, sanitized_tag)
            try:
                os.makedirs(tag_dir, exist_ok=True)
            except OSError as e:
                self.logger.error(f"Failed create CSV dir '{tag_dir}': {e}")
                continue
            csv_path = os.path.join(tag_dir, f"summary_{sanitized_tag}_{datetime.now().strftime('%Y%m%d')}.csv")
            csv_data = []
            try:
                async with self.tracking_lock:
                    cursor = self.db_conn.cursor()
                    cursor.execute('''SELECT T1.path, T1.url, T1.checkpoint_name, T2_All.tag FROM tracked_images T1 JOIN image_tags T_Current ON T1.image_key = T_Current.image_key AND T_Current.tag = ? LEFT JOIN image_tags T2_All ON T1.image_key = T2_All.image_key AND T2_All.tag != ?''', (tag, tag))
                    rows = cursor.fetchall()
                processed = {}
                for path, url, cp_name, other_tag in rows:
                    if path not in processed:
                        processed[path] = {"cp": cp_name or "Unknown", "url": url or "N/A", "others": set()}
                    if other_tag:
                        processed[path]["others"].add(other_tag)
                for img_path, data in processed.items():
                    rel_path = img_path
                    if img_path and os.path.exists(tag_dir) and os.path.exists(img_path):
                        try:
                            rel_path = os.path.relpath(img_path, tag_dir)
                        except ValueError:
                            pass
                    others_list = sorted(list(data["others"]))
                    if others_list:
                        csv_data.extend([[tag, ot, data["cp"], rel_path, data["url"]] for ot in others_list])
                    else:
                        csv_data.append([tag, "", data["cp"], rel_path, data["url"]])
            except sqlite3.Error as e:
                self.logger.error(f"DB query error for tag '{tag}': {e}")
                continue
            if not csv_data:
                self.logger.info(f"No images found for tag '{tag}' summary.")
                continue
            try:
                with open(csv_path, "w", newline="", encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(["Current Tag", "Previously Downloaded Tag", "Checkpoint Name", "Relative Image Path", "Download URL"])
                    w.writerows(csv_data)
                self.logger.info(f"Wrote summary CSV: {csv_path}")
            except Exception as e:
                self.logger.error(f"Failed write CSV for '{tag}': {e}", exc_info=True)


# ===================
# --- Entry Point ---
# ===================
if __name__ == "__main__":
    cli_args = parse_arguments()
    is_cli = cli_args.mode is not None
    print(f"--- Civitai Downloader v{SCRIPT_VERSION} ---")
    print(f"Running in {'command-line' if is_cli else 'interactive'} mode.")
    logger.info(f"Running in {'command-line' if is_cli else 'interactive'} mode.")
    try:
        downloader = CivitaiDownloader(cli_args)
        asyncio.run(downloader.run())
    except SystemExit:
        logger.warning("Exiting (SystemExit).")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (Ctrl+C).")
        print("\nProcess interrupted.")
    except Exception as main_err:
        logger.critical(f"Unhandled exception at main level: {main_err}", exc_info=True)
        print(f"\n--- UNHANDLED CRITICAL ERROR ---\nError: {main_err}\nCheck log: {log_file_path}\n----------------------")
    logger.info("Script finished.")
    print("\nDownload process complete.")
