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

# Color Code
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# --- Tenacity for Retries ---
try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception, before_sleep_log, RetryError
    from tenacity.stop import stop_base
except ImportError:
    print("Error: 'tenacity' library not found. Please install it using: pip install tenacity")
    sys.exit(1)

# --- anext compatibility ---
try:
    from asyncio import anext # Python 3.10+
except ImportError:
    _MISSING = object()
    async def anext(ait: AsyncGenerator, default: Any = _MISSING) -> Any:
        try: return await ait.__anext__()
        except StopAsyncIteration:
            if default is _MISSING: raise
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
DATABASE_FILENAME: str = "tracking_database.sqlite" # <--- SQLite DB filename
LOG_FILENAME_TEMPLATE: str = "civit_image_downloader_log_{version}.txt"
SCRIPT_VERSION: str = "1.3-sqlite" 
DEFAULT_TIMEOUT: int = 60
DEFAULT_RETRIES: int = 2
DEFAULT_MAX_PATH_LENGTH: int = 240

# Global retry count (set before creating downloader instance)
CURRENT_RETRY_COUNT: int = DEFAULT_RETRIES

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
RETRYABLE_EXCEPTIONS: Tuple[type[Exception], ...] = ( httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, ConnectionResetError )
RETRYABLE_STATUS_CODES: set[int] = { 429, 500, 502, 503, 504 }  # Added 429 for rate limiting

def is_retryable_http_status(exception: BaseException) -> bool:
    return isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code in RETRYABLE_STATUS_CODES

def should_retry_exception(exception: BaseException) -> bool:
    return isinstance(exception, RETRYABLE_EXCEPTIONS) or is_retryable_http_status(exception)

# Custom stop condition that uses global retry count
class stop_after_dynamic_attempt(stop_base):
    """Stop after a dynamically determined number of attempts."""
    def __call__(self, retry_state):
        return retry_state.attempt_number > (1 + CURRENT_RETRY_COUNT)

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
    parser.add_argument("--disable_prompt_check", choices=['y', 'n'], default='n', help="Disable prompt check in Mode 3 and Mode 4 (y/n).")
    parser.add_argument("--username", help="Username(s) for Mode 1 (comma-separated).")
    parser.add_argument("--model_id", help="Model ID(s) for Mode 2 (comma-separated, numeric).")
    parser.add_argument("--model_version_id", help="Model Version ID(s) for Mode 4 (comma-separated, numeric).")
    parser.add_argument("--filter_tags", help="Filter by tag(s) in Mode 4 (comma-separated). Only downloads images matching these tags.")
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
    if data.startswith(b'\x89PNG\r\n\x1a\n'): return ".png"
    if data.startswith(b'\xff\xd8\xff'): return ".jpeg"
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP': return ".webp"
    if len(data) >= 8 and data[4:8] == b'ftyp': return ".mp4"
    if data.startswith(b'\x1A\x45\xDF\xA3'): return ".webm"
    return None

def extract_image_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts the actual metadata dict from an API item, handling nested structure.
    
    CivitAI API changed structure from:
        item["meta"] = {"prompt": "...", "Model": "..."}
    To:
        item["meta"] = {"id": 123, "meta": {"prompt": "...", "Model": "..."}}
    
    This function handles both old and new structures.
    """
    meta_field = item.get("meta")
    if not meta_field or not isinstance(meta_field, dict):
        return {}
    
    # Check for new nested structure: meta.meta exists and contains generation params
    nested_meta = meta_field.get("meta")
    if nested_meta and isinstance(nested_meta, dict):
        # New structure: actual metadata is in meta.meta
        return nested_meta
    
    # Old structure or meta doesn't have nested meta: check if prompt/Model exists at top level
    if "prompt" in meta_field or "Model" in meta_field or "seed" in meta_field:
        return meta_field
    
    # Fallback: return empty dict if no recognizable structure
    return {}

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
        self.tracking_lock: Lock = Lock() # Lock for database write operations
        self.tag_model_mapping: Dict[str, List[Tuple[int, str]]] = {}
        self.tag_model_mapping_lock: Lock = Lock()
        self.visited_api_urls: set[str] = set()
        self.run_results: Dict[str, Dict[str, Any]] = {}
        self.skipped_reasons_summary: Dict[str, int] = {}
        self.failed_urls: List[str] = []
        self.failed_search_requests: List[str] = []

        # --- Log Initialized Config ---
        try: os.makedirs(self.output_dir, exist_ok=True)
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
            cursor = self.db_conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;") # Enable FK constraints
            # Create tracked_images table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracked_images (
                    image_key TEXT PRIMARY KEY, image_id TEXT NOT NULL, quality TEXT NOT NULL,
                    path TEXT NOT NULL, download_date TEXT NOT NULL, url TEXT, checkpoint_name TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracked_images_key ON tracked_images (image_key)')
            # Create image_tags table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS image_tags (
                    image_key TEXT NOT NULL, tag TEXT NOT NULL,
                    PRIMARY KEY (image_key, tag),
                    FOREIGN KEY(image_key) REFERENCES tracked_images(image_key) ON DELETE CASCADE
                ) WITHOUT ROWID;
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags (tag)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_tags_key ON image_tags (image_key)')
            self.db_conn.commit()
            self.logger.debug("Database schema initialized successfully (Relational Tags).")
        except sqlite3.Error as e:
            self.logger.critical(f"Database error during initialization: {e}", exc_info=True)
            print(f"CRITICAL ERROR: Failed to initialize tracking database '{self.db_path}'.")
            if self.db_conn:
                 try: self.db_conn.close()
                 except sqlite3.Error: pass
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

        # Check if a value was provided via the command line
        cli_value_provided = self.args.timeout is not None and self.args.timeout != DEFAULT_TIMEOUT

        # Check if the *provided* CLI value is valid (>0)
        cli_value_is_valid = cli_value_provided and self.args.timeout > 0

        if cli_value_provided:
            if cli_value_is_valid:
                # A valid value (potentially non-default or the default itself) was explicitly given
                self.logger.warning(f"Timeout set via CLI argument: {self.args.timeout}")
                return self.args.timeout
            else:
                # An invalid value was given via CLI, log error and use default
                self.logger.warning(f"Invalid --timeout value '{self.args.timeout}' provided via CLI. Using default: {DEFAULT_TIMEOUT}s.")
                return DEFAULT_TIMEOUT
        # ---- If NO value was provided via CLI ----
        elif self._is_interactive_mode():
             # Only prompt if interactive AND no CLI value was given
             self.logger.debug("Interactive mode detected for timeout, prompting user...")
             while True:
                try:
                    timeout_input = input(f"Enter timeout value in seconds [default: {DEFAULT_TIMEOUT}]: ").strip()
                    if not timeout_input: # User hit Enter for default
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
            # Not interactive mode, and no value provided via CLI. Use default.
            self.logger.debug(f"Non-interactive mode, using default timeout: {DEFAULT_TIMEOUT}")
            return DEFAULT_TIMEOUT

    def _get_quality(self) -> str:
        if self.args.quality == 2: return 'HD'
        if self.args.quality == 1: return 'SD'
        elif self._is_interactive_mode():
            while True:
                inp = input("Choose quality (1=SD, 2=HD) [default: 1]: ").strip()
                if inp == '2': return 'HD'
                if inp == '1' or inp == '': return 'SD'
                print("Invalid choice.")
        else: return 'SD' 

    def _get_redownload_option(self) -> int:
        if self.args.redownload == 1: return 1 
        elif self._is_interactive_mode():
            self.logger.debug("Prompting user for redownload option...")
            while True:
                inp = input("Allow re-downloading tracked items? (1=Yes, 2=No) [default: 2]: ").strip()
                self.logger.debug(f"User input for redownload: '{inp}'")
                if inp == '1': self.logger.debug("User selected redownload: Yes"); return 1
                if inp == '2' or inp == '': self.logger.debug("User selected redownload: No"); return 2
                print("Invalid choice.")
        else: return self.args.redownload # Use default (2) if not interactive and not explicitly 1

    def _get_mode_choice(self) -> Optional[str]:
        if self.args.mode in [1, 2, 3, 4]: return str(self.args.mode)
        elif self._is_interactive_mode():
            while True:
                inp = input("Choose mode (1=user, 2=model ID, 3=tag search, 4=model version ID): ").strip()
                if inp in ['1', '2', '3', '4']: return inp
                print("Invalid choice.")
        else: 
            self.logger.error("Operation mode (--mode) is required in non-interactive mode.")
            print("Error: Operation mode (--mode) is required.")
            return None

    def _get_semaphore_limit(self) -> int:
        is_cli_non_default = (self.args.semaphore_limit is not None and
                              self.args.semaphore_limit != DEFAULT_SEMAPHORE_LIMIT and
                              self.args.semaphore_limit > 0)
        if is_cli_non_default: return self.args.semaphore_limit
        elif self._is_interactive_mode():
             self.logger.debug("Prompting user for semaphore limit...")
             while True:
                try:
                    inp = input(f"Enter max concurrent downloads [default: {DEFAULT_SEMAPHORE_LIMIT}]: ").strip()
                    if not inp: self.logger.info(f"Using default semaphore: {DEFAULT_SEMAPHORE_LIMIT}"); return DEFAULT_SEMAPHORE_LIMIT
                    val = int(inp); assert val > 0; self.logger.info(f"User set semaphore limit: {val}"); return val
                except (ValueError, AssertionError): print("Invalid input. Must be positive number.")
        else: # Not interactive, use default or validated arg value
            valid_val = self.args.semaphore_limit if self.args.semaphore_limit is not None and self.args.semaphore_limit > 0 else DEFAULT_SEMAPHORE_LIMIT
            self.logger.debug(f"Using default/parsed semaphore limit: {valid_val}")
            return valid_val

    # --- Result Dictionary Access Helper ---
    def _get_result_entry(self, parent_key: Optional[str], model_id: Optional[int] = None) -> Optional[Dict]:
         """Helper to get the correct dictionary entry in run_results to update stats."""
         if not parent_key: return None # Cannot update if no key provided
         if parent_key not in self.run_results:
              self.logger.error(f"Attempted to get result entry for non-existent key: {parent_key}")
              return None
         if model_id is not None: # Tag mode sub-detail
              model_key = f"model:{model_id}"
              if 'sub_details' not in self.run_results[parent_key]: self.run_results[parent_key]['sub_details'] = {}
              if model_key not in self.run_results[parent_key]['sub_details']:
                   self.run_results[parent_key]['sub_details'][model_key] = { 'success_count': 0, 'skipped_count': 0, 'no_meta_count': 0, 'api_items': 0, 'status': 'Pending', 'reason': None }
              return self.run_results[parent_key]['sub_details'][model_key]
         else: # Direct identifier mode
              return self.run_results[parent_key]

    # --- Basic Client-Side Validation ---
    async def _validate_identifier_basic(self, identifier: str, id_type: str) -> Tuple[bool, Optional[str]]:
        """Performs basic client-side validation. Returns (is_valid, reason_if_invalid)."""
        if id_type in ['model', 'modelVersion']:
            if not identifier.isdigit() or int(identifier) <= 0:
                return False, f"Identifier '{identifier}' must be a positive number for type '{id_type}'."
        elif id_type == 'username':
            if not identifier or identifier.isspace(): return False, "Username cannot be empty."
        elif id_type == 'tag':
            if not identifier or identifier.isspace(): return False, "Tag cannot be empty."
        return True, None

    # --- SQLite Tracking Methods ---
    async def check_if_image_downloaded(self, image_id: str, quality: str, context: Optional[str] = None) -> bool:
        """Checks if an image exists in the tracking database.

        Args:
            image_id: The image ID
            quality: Image quality (SD/HD)
            context: Optional context string in format "type:identifier" (e.g., "username:Exorvious")
                     to prevent cross-contamination between different query types.
        """
        if not self.db_conn: return False # Assume not downloaded if DB error
        # Include context in key to prevent cross-contamination (Bug #47 fix)
        if context:
            image_key = f"{context}_{str(image_id)}_{quality}"
        else:
            # Fallback for backwards compatibility
            image_key = f"{str(image_id)}_{quality}"
        query = "SELECT 1 FROM tracked_images WHERE image_key = ? LIMIT 1"
        try:
            # Use lock to prevent TOCTOU race conditions with concurrent checks
            async with self.tracking_lock:
                cursor = self.db_conn.cursor()
                cursor.execute(query, (image_key,))
                exists = cursor.fetchone() is not None
                return exists
        except sqlite3.Error as e:
            self.logger.error(f"DB error checking if image {image_key} downloaded: {e}", exc_info=True)
            return False # Assume not downloaded on error

    async def mark_image_as_downloaded(self, image_id: str, image_path: str, quality: str, tags: Optional[List[str]] = None, url: Optional[str] = None, checkpoint_name: Optional[str] = None, context: Optional[str] = None) -> None:
        """Marks image as downloaded in DB (INSERT OR REPLACE image, DELETE/INSERT tags).

        Args:
            image_id: The image ID
            image_path: Path where the image was saved
            quality: Image quality (SD/HD)
            tags: Optional list of tags
            url: Optional source URL
            checkpoint_name: Optional model/checkpoint name
            context: Optional context string in format "type:identifier" (e.g., "username:Exorvious")
                     to prevent cross-contamination between different query types.
        """
        if not self.db_conn: return
        image_id_str = str(image_id)
        # Include context in key to prevent cross-contamination (Bug #47 fix)
        if context:
            image_key = f"{context}_{image_id_str}_{quality}"
        else:
            # Fallback for backwards compatibility
            image_key = f"{image_id_str}_{quality}"
        current_date = datetime.now().strftime("%Y-%m-%d - %H:%M")
        tags = tags or []
        unique_tags = sorted(list(set(t for t in tags if t))) # Ensure unique, non-empty

        async with self.tracking_lock: # Lock for the entire transaction
             cursor = self.db_conn.cursor()
             try:
                 # 1. Upsert main image data
                 cursor.execute('''
                     INSERT OR REPLACE INTO tracked_images
                     (image_key, image_id, quality, path, download_date, url, checkpoint_name)
                     VALUES (?, ?, ?, ?, ?, ?, ?)
                 ''', (image_key, image_id_str, quality, image_path, current_date, url, checkpoint_name))
                 # 2. Delete existing tags for this image
                 cursor.execute('DELETE FROM image_tags WHERE image_key = ?', (image_key,))
                 # 3. Insert new tags if any
                 if unique_tags:
                      tag_insert_data = [(image_key, tag) for tag in unique_tags]
                      cursor.executemany('INSERT INTO image_tags (image_key, tag) VALUES (?, ?)', tag_insert_data)
                 # 4. Commit
                 self.db_conn.commit()
                 self.logger.debug(f"Marked downloaded (in DB): {image_id_str} ({quality}) -> {image_path} (Tags: {len(unique_tags)})")
             except sqlite3.Error as e:
                 self.logger.error(f"Database error marking image {image_key} as downloaded: {e}", exc_info=True)
                 try: self.db_conn.rollback()
                 except sqlite3.Error as rb_e: self.logger.error(f"Rollback failed: {rb_e}")


        # ===================================
        # --- Core Download/File Methods ---
        # ===================================
    @retry(
        stop=stop_after_dynamic_attempt(),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception(should_retry_exception),
        before_sleep=before_sleep_log(retry_logger, logging.WARNING)
    )
    async def download_image(self, image_api_item: Dict[str, Any], base_output_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Downloads a single image, detects extension, saves, with retries."""
        image_url = image_api_item.get('url')
        image_id = image_api_item.get('id')
        # Basic validation
        if not image_url or not image_id: return False, None, "Missing URL or ID in API data"
        if not base_output_path or not os.path.isdir(base_output_path): return False, None, f"Invalid target directory '{base_output_path}'"

        # Prepare URL and path
        target_url = image_url
        if self.quality == 'HD':
             target_url = re.sub(r"width=\d{3,4}", "original=true", image_url)
             if target_url == image_url: target_url += ('&' if '?' in target_url else '?') + "original=true"
             self.logger.debug(f"HD URL: {target_url}")
        final_image_path = None

        try:
            async with self.semaphore:
                client = await self._get_client()
                async with client.stream("GET", target_url) as response:
                    if 400 <= response.status_code < 500: return False, None, f"HTTP Client Error {response.status_code} {response.reason_phrase}" # Not retryable
                    response.raise_for_status() # Raises for >=400. Retry logic catches 5xx.

                    # Extension Detection
                    total_size = int(response.headers.get('content-length', 0))
                    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                    mapping = { "image/png": ".png", "image/jpeg": ".jpeg", "image/jpg": ".jpeg", "image/webp": ".webp", "video/mp4": ".mp4", "video/webm": ".webm" }
                    file_extension = mapping.get(content_type)
                    byte_iter = response.aiter_bytes()
                    first_chunk = await anext(byte_iter, None)
                    if first_chunk is None: return False, None, "Downloaded empty file"
                    if file_extension is None: file_extension = detect_extension(first_chunk)
                    if file_extension is None: file_extension = ".png" if self.quality == 'HD' else ".jpeg"

                    # Path Construction
                    filename_base = self._clean_path_component(str(image_id))
                    potential_final_path = os.path.join(base_output_path, filename_base + file_extension)
                    if len(potential_final_path) > self.max_path_length:
                        allowed_len = self.max_path_length - len(base_output_path) - 1
                        if allowed_len < 10: return False, None, "Path too long, cannot shorten"
                        shortened_filename = self._clean_path_component(filename_base + file_extension, max_length=allowed_len)
                        final_image_path = os.path.join(base_output_path, shortened_filename)
                    else: final_image_path = potential_final_path
                    final_dir = os.path.dirname(final_image_path)
                    if not final_dir: return False, None, "Invalid final path structure"

                    # File Writing
                    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"DL {os.path.basename(final_image_path)}", leave=False, dynamic_ncols=True)
                    downloaded_size = 0
                    try:
                        os.makedirs(final_dir, exist_ok=True)
                        async with aiofiles.open(final_image_path, "wb") as file:
                             if first_chunk:
                                 await file.write(first_chunk)
                                 progress_bar.update(len(first_chunk)); downloaded_size += len(first_chunk)
                             async for chunk in byte_iter:
                                 await file.write(chunk)
                                 progress_bar.update(len(chunk)); downloaded_size += len(chunk)
                    finally: progress_bar.close()

                    # Final Checks
                    if total_size != 0 and downloaded_size < total_size:
                         try: os.remove(final_image_path); 
                         except OSError: pass
                         return False, None, "Incomplete download"
                    self.logger.debug(f"Successfully downloaded: {final_image_path}")
                    return True, final_image_path, None

        # Exception Handling (Final/Non-Retryable)
        except RetryError as e:
            reason = f"Max retries exceeded: {e}"
            self.logger.error(f"Download failed for {target_url} after retries: {e}")
            return False, None, reason

        except httpx.RequestError as e:
            reason = f"Network error: {e.__class__.__name__}"
            self.logger.error(f"Network error DL {target_url} (final): {e}")
            return False, None, reason

        except httpx.HTTPStatusError as e:
            reason = f"HTTP Error {e.response.status_code}"
            self.logger.error(f"HTTP error DL {target_url} (final): {e}")
            return False, None, reason

        except OSError as e:
            reason = f"File system error: {e.__class__.__name__}"
            self.logger.error(f"FS error DL/write {target_url}: {e}", exc_info=True)
            return False, None, reason

        except Exception as e:
            reason = f"Unexpected error: {e.__class__.__name__}"
            self.logger.critical(f"Unexpected error DL {target_url}: {e}", exc_info=True)

            return False, None, reason

        finally:
            if final_image_path and not os.path.exists(final_image_path):
                partial_file_path_to_check = final_image_path + ".partial"
                if os.path.exists(partial_file_path_to_check):
                     try:
                          os.remove(partial_file_path_to_check)
                     except OSError:
                          pass
                      
    async def _write_meta_data(self, meta: Optional[Dict[str, Any]], base_output_path_no_ext: str, image_id: str, username: Optional[str], base_model: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Writes metadata to a .txt file.

        Args:
            meta: Metadata dictionary extracted from API response
            base_output_path_no_ext: Base path for output file (without extension)
            image_id: Image ID
            username: Username associated with the image
            base_model: Optional baseModel from API (e.g., "Flux.1 D", "SDXL 1.0") to use
                        when Model field is missing in metadata (Bug #38 fix)
        """
        username = username or 'unknown_user'; meta = meta or {}
        content_lines = []; meta_filename_suffix = ""
        if not meta or all(str(v).strip() == '' for v in meta.values()):
             meta_filename_suffix = "_no_meta.txt"
             content_lines = ["No metadata available.", f"URL: https://civitai.com/images/{image_id}?username={username}"]
        else:
             meta_filename_suffix = "_meta.txt"
             # Bug #38 fix: Add Model field from baseModel if missing
             if base_model and 'Model' not in meta:
                 # Insert Model at the beginning for consistency with old metadata format
                 content_lines = [f"Model: {base_model}"]
                 content_lines.extend([f"{k}: {str(v) if v is not None else ''}" for k, v in meta.items()])
             else:
                 content_lines = [f"{k}: {str(v) if v is not None else ''}" for k, v in meta.items()]

        directory = os.path.dirname(base_output_path_no_ext)
        base_filename = os.path.basename(base_output_path_no_ext)
        meta_filename = base_filename + meta_filename_suffix
        try:
             max_fname_len = self.max_path_length - len(directory) - 1
             if max_fname_len < 10: raise ValueError("Base directory path too long")
             cleaned_meta_filename = self._clean_path_component(meta_filename, max_length=max_fname_len)
             output_path_final = os.path.join(directory, cleaned_meta_filename)
        except Exception as path_e:
             self.logger.error(f"Error constructing metadata path for {base_filename}: {path_e}"); return False, None

        try:
             if not directory: raise ValueError("Cannot determine directory")
             os.makedirs(directory, exist_ok=True)
             async with aiofiles.open(output_path_final, "w", encoding='utf-8') as f: await f.write("\n".join(content_lines)); await f.flush()
             self.logger.debug(f"Wrote metadata: {output_path_final}"); return True, output_path_final
        except (OSError, ValueError, Exception) as e:
             self.logger.error(f"Error writing metadata to {output_path_final}: {e}", exc_info=True); return False, None

    # --- API Fetching Method ---
    @retry(
        stop=stop_after_dynamic_attempt(),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception(should_retry_exception),
        before_sleep=before_sleep_log(retry_logger, logging.WARNING)
    )
    async def _fetch_api_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetches a single page from the Civitai images API, with retries."""
        if url in self.visited_api_urls: return None
        self.visited_api_urls.add(url)
        client = await self._get_client(); self.logger.debug(f"Fetching API page: {url}")
        try:
            async with self.semaphore: response = await client.get(url)

            # Handle non-retryable client errors (4xx)
            if 400 <= response.status_code < 500:
                if response.status_code == 404: self.logger.warning(f"API request returned 404 Not Found for {url}"); return None
                else: self.logger.error(f"API Client Error {response.status_code} for {url} (No Retry)"); self.failed_urls.append(url); return None

            # --- Specific Handling for 500 Errors ---
            if response.status_code == 500:
                 try:
                     data = response.json()
                     # Check for specific error messages that indicate identifier not found
                     error_msg = data.get('error', '').lower()
                     message_msg = data.get('message', '').lower() # Some errors might be in 'message'
                     if 'user not found' in error_msg:
                          self.logger.warning(f"API reported 'User not found' for {url} (Status 500)")
                          return data # <-- Return the data containing the error message
                     if 'model not found' in error_msg or 'model not found' in message_msg:
                          self.logger.warning(f"API reported 'Model not found' for {url} (Status 500)")
                          return data # <-- Return data for model not found
                     if 'version not found' in error_msg or 'version not found' in message_msg:
                          self.logger.warning(f"API reported 'Version not found' for {url} (Status 500)")
                          return data # <-- Return data for version not found

                     # If it's a 500 but not a known "not found" error, raise for retry
                     self.logger.warning(f"API returned generic 500 Server Error for {url}. Raising for potential retry.")
                     response.raise_for_status() # Raise the 500 error to trigger tenacity retry

                 except json.JSONDecodeError:
                      # If 500 response isn't valid JSON, raise for retry
                      self.logger.warning(f"API returned 500 with invalid JSON for {url}. Raising for potential retry.")
                      response.raise_for_status() # Raise the original 500 error

            # --- End Specific Handling for 500 ---

            # Raise for other retryable (5xx not handled above) or unexpected status codes
            response.raise_for_status()

            # Success (2xx)
            try: return response.json()
            except json.JSONDecodeError as e: self.logger.error(f"JSON Decode Error {url} (Status {response.status_code}): {e}"); self.failed_urls.append(url); return None

        # --- Exception Handling (Final/Non-Retryable) ---
        except RetryError as e:
            self.logger.error(f"API fetch failed after retries {url}: {e}")
            self.failed_urls.append(url)
            return None
        
        except httpx.RequestError as e:
            self.logger.error(f"Network error fetch {url} (final): {e}")
            self.failed_urls.append(url)
            return None
        
        except httpx.HTTPStatusError as e: # Catches final 5xx or non-retryable HTTP errors raised above
            self.logger.error(f"HTTP error fetch {url} (final): Status {e.response.status_code}")
            self.failed_urls.append(url)
            return None
        
        except Exception as e:
            self.logger.critical(f"Unexpected error fetch {url}: {e}", exc_info=True)
            self.failed_urls.append(url)
            return None
        
    # --- Mode Execution Logic ---
    async def _process_api_items(self, items: List[Dict[str, Any]], target_dir: str, mode_specific_info: Optional[Dict] = None, parent_result_key: Optional[str] = None, model_id: Optional[int] = None) -> None:
        """Processes image items: checks skip conditions, creates download tasks."""
        tasks = []; mode_specific_info = mode_specific_info or {}
        tag_to_check = mode_specific_info.get('tag_to_check'); disable_prompt_check = mode_specific_info.get('disable_prompt_check', False); current_tag = mode_specific_info.get('current_tag')
        # DEBUG: Log first item structure to understand API response (only if debug enabled)
        if self.logger.isEnabledFor(logging.DEBUG) and items and len(items) > 0:
            first_item = items[0]
            self.logger.debug(f"DEBUG - First item keys: {list(first_item.keys())}")
            raw_meta = first_item.get("meta")
            self.logger.debug(f"DEBUG - Raw meta field type: {type(raw_meta)}")
            extracted_meta = extract_image_meta(first_item)
            self.logger.debug(f"DEBUG - Extracted meta keys: {list(extracted_meta.keys()) if extracted_meta else 'None'}")
            if extracted_meta:
                prompt_preview = extracted_meta.get("prompt", "")[:100]
                self.logger.debug(f"DEBUG - Prompt preview: {prompt_preview}...")
        for item in items:
            image_id = item.get('id'); 
            if not image_id: continue
            should_skip, skip_reason = False, None
            result_entry = self._get_result_entry(parent_result_key, model_id)
            # Redownload Check
            if self.allow_redownload == 2 and await self.check_if_image_downloaded(str(image_id), self.quality, context=parent_result_key):
                skip_reason = "Already tracked"; should_skip = True
                if current_tag: # Update tags in DB if needed? Requires SELECT+UPDATE - skip for now. Pass.
                    pass # Simpler: Don't update tags on skipped items via this path.
            # Tag Prompt Check
            if not should_skip and tag_to_check and not disable_prompt_check:
                 meta = extract_image_meta(item)
                 prompt = meta.get("prompt", "").lower()
                 if not all(word in prompt for word in tag_to_check.lower().split("_") if word):
                     skip_reason = f"Prompt check failed: {tag_to_check}"; should_skip = True
            # Update stats if skipped
            if should_skip:
                if result_entry: result_entry['skipped_count'] += 1
                if skip_reason: self.skipped_reasons_summary[skip_reason] = self.skipped_reasons_summary.get(skip_reason, 0) + 1
                continue
            # Create Download Task if not skipped
            tasks.append(self._handle_single_download(item, target_dir, current_tag, parent_result_key, model_id))
        if tasks: await asyncio.gather(*tasks) # Run downloads for this batch

    async def _handle_single_download(self, item: Dict[str, Any], target_dir: str, current_tag: Optional[str] = None, parent_result_key: Optional[str] = None, model_id: Optional[int] = None) -> None:
        """Handles download, meta write, tracking, stats update for one item."""
        image_id = item.get('id'); 
        if not image_id: return
        base_path_no_ext = os.path.join(target_dir, self._clean_path_component(str(image_id)))
        result_entry = self._get_result_entry(parent_result_key, model_id)
        success, final_image_path, reason = await self.download_image(item, target_dir)
        if success and final_image_path:
             meta = extract_image_meta(item); username = item.get("username")
             # Bug #38 fix: Extract model name from civitaiResources or baseModel
             model_name_for_meta = None
             checkpoint_name = str(meta.get("Model", "")).strip() if meta and meta.get("Model") else None

             # Bug #42 fix: Detect and skip URN-format Model names (e.g., "urn_air_sdxl_checkpoint_civitai_101055@128078")
             # These are ComfyUI-style resource identifiers, not human-readable model names
             if checkpoint_name and checkpoint_name.lower().startswith("urn") and ("civitai" in checkpoint_name.lower() or "@" in checkpoint_name):
                 self.logger.debug(f"Model field contains URN format ({checkpoint_name[:50]}...), will extract from civitaiResources instead")
                 checkpoint_name = None  # Force extraction from civitaiResources

             # If no Model in meta (or was URN format), try to extract from civitaiResources
             if not checkpoint_name and meta:
                 civitai_resources = meta.get("civitaiResources", [])
                 if isinstance(civitai_resources, list):
                     # Look for checkpoint type resource
                     for resource in civitai_resources:
                         if isinstance(resource, dict) and resource.get("type") == "checkpoint":
                             model_name_for_meta = resource.get("modelVersionName")
                             checkpoint_name = model_name_for_meta
                             break

                 # Fallback to baseModel only if no checkpoint found in civitaiResources
                 if not model_name_for_meta:
                     base_model = item.get("baseModel")
                     if base_model:
                         model_name_for_meta = base_model
                         checkpoint_name = base_model
             if result_entry: result_entry['success_count'] += 1
             tags_to_mark = [current_tag] if current_tag else []
             await self.mark_image_as_downloaded(str(image_id), final_image_path, self.quality, tags=tags_to_mark, url=item.get('url'), checkpoint_name=checkpoint_name, context=parent_result_key)
             await self._write_meta_data(meta, base_path_no_ext, str(image_id), username, base_model=model_name_for_meta)
             if not meta or all(str(v).strip() == '' for v in meta.values()):
                 if result_entry: result_entry['no_meta_count'] += 1
        elif not success:
             if result_entry: result_entry['skipped_count'] += 1
             fail_reason = reason or "Download failed"
             self.skipped_reasons_summary[fail_reason] = self.skipped_reasons_summary.get(fail_reason, 0) + 1

    async def _run_paginated_download(self, initial_url: str, target_dir: str, mode_specific_info: Optional[Dict] = None, parent_result_key: Optional[str] = None, model_id: Optional[int] = None) -> None:
        """Handles pagination and processing for an identifier, checking for 'Not Found' errors."""
        url: Optional[str] = initial_url; page_count: int = 0; identifier_status: str = 'Pending'
        identifier_reason: Optional[str] = None; identifier_api_items: int = 0
        result_entry = self._get_result_entry(parent_result_key, model_id)
        if not result_entry: self.logger.error(f"Result entry missing for {parent_result_key}/{model_id}"); return

        # Bug #50 enhancement: Print status message when starting to process an identifier
        identifier_display = parent_result_key if not model_id else f"{parent_result_key} → model_{model_id}"
        print(f"\n{'='*60}\nStarting: {identifier_display}\n{'='*60}")

        while url:
             page_count += 1
             self.logger.info(f"Requesting API page {page_count} for {os.path.basename(target_dir)}")
             page_data = await self._fetch_api_page(url) # Retries handled inside

             # --- Add Check for Specific "Not Found" Errors on First Page ---
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
                      # Update result entry immediately and stop processing this identifier
                      result_entry['status'] = identifier_status
                      result_entry['reason'] = identifier_reason
                      # Print specific message to console
                      print(f"Error for '{parent_result_key}': {identifier_reason}. Please check the identifier.")
                      return # Stop processing this identifier completely
             # --- End "Not Found" Check ---

             if page_data is not None: # Process page if fetch succeeded and wasn't a "Not Found" error handled above
                 items = page_data.get('items', [])
                 metadata = page_data.get('metadata', {})
                 item_count = len(items); identifier_api_items += item_count
                 self.logger.info(f"Processing {item_count} items from page {page_count}")

                 if items:
                      if identifier_status == 'Pending': identifier_status = 'Processing'
                      await self._process_api_items(items, target_dir, mode_specific_info, parent_result_key, model_id)

                      # Bug #50 enhancement: Print progress update after processing each page
                      current_downloads = result_entry.get('success_count', 0)
                      current_skipped = result_entry.get('skipped_count', 0)
                      print(f"[{identifier_display}] Page {page_count}: Downloaded {current_downloads} | Skipped {current_skipped} | API items processed: {identifier_api_items}")
                 elif page_count == 1: # No items on the first page (and not a "Not Found" error)
                      identifier_status = 'Completed (No Items Found)'
                      self.logger.warning(f"No items found for {os.path.basename(target_dir)} (User/Model/Version may have no images).")
                      # Also print this info to console for clarity
                      print(f"Info for '{parent_result_key}': Identifier found, but no images associated with it.")


                 url = metadata.get('nextPage')
                 if not url: # No more pages
                      if identifier_status != 'Failed (Not Found)': # Avoid overwriting specific failure
                         if identifier_status == 'Processing': identifier_status = 'Completed'
                         elif identifier_status == 'Pending': identifier_status = 'Completed (No Items Found)'
                      self.logger.debug(f"No next page found for {os.path.basename(target_dir)}."); break
                 else: await asyncio.sleep(1)
             else: # Fetch failed or returned 404 etc.
                 fetch_fail_reason = f"Failed to fetch API page {page_count}"
                 if url in self.failed_urls: fetch_fail_reason += " (check logs for specific URL error)"
                 if page_count == 1: identifier_status = 'Failed'; identifier_reason = fetch_fail_reason
                 else: identifier_status = 'Completed (Fetch Error on Subsequent Page)'; identifier_reason = fetch_fail_reason
                 self.logger.warning(f"Stopping pagination for {os.path.basename(target_dir)}: {fetch_fail_reason}."); break

        # Update final status (unless already set to Failed Not Found)
        if result_entry and identifier_status != 'Failed (Not Found)':
            result_entry['status'] = identifier_status
            if identifier_reason and not result_entry.get('reason'): result_entry['reason'] = identifier_reason # Don't overwrite specific "Not Found" reason
            result_entry['api_items'] = identifier_api_items
            # Aggregate API items
            if model_id is not None and parent_result_key in self.run_results: self.run_results[parent_result_key]['api_items'] += identifier_api_items

        # Sorting
        if not self.disable_sorting and identifier_status.startswith('Completed'):
            self.logger.info(f"Running sorting for: {target_dir}"); await self._sort_images_by_model_name(target_dir)

        # Bug #50 enhancement: Print completion message with summary
        total_downloads = result_entry.get('success_count', 0)
        total_skipped = result_entry.get('skipped_count', 0)
        total_api_items = result_entry.get('api_items', 0)
        print(f"\n{'='*60}\nCompleted: {identifier_display}")
        print(f"Status: {identifier_status}")
        print(f"Downloaded: {total_downloads} | Skipped: {total_skipped} | API items: {total_api_items}")
        print(f"{'='*60}\n")

    # --- Tag Search Methods ---
    @retry(stop=stop_after_dynamic_attempt(), wait=wait_random_exponential(multiplier=1, max=10), retry=retry_if_exception(should_retry_exception), before_sleep=before_sleep_log(retry_logger, logging.WARNING))
    async def _search_models_by_tag_page(self, url: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        """Helper: Fetches one page of model search results, with retries."""
        self.logger.debug(f"Fetching models page: {url}")
        async with self.semaphore: response = await client.get(url)
        if 400 <= response.status_code < 500: self.logger.error(f"Model search Client Error {response.status_code} for {url}"); self.failed_search_requests.append(url); return None
        response.raise_for_status()
        try: return response.json()
        except json.JSONDecodeError as e: self.logger.error(f"JSON Decode Error model search {url}: {e}"); self.failed_search_requests.append(url); return None

    async def _search_models_by_tag(self, tag_query: str) -> List[Tuple[int, str]]:
        """
        Searches for model IDs and names by tag, handling pagination and adding
        validation based on first page results and a maximum page limit.
        """
        encoded_tag = tag_query.replace(" ", "%20")
        url: Optional[str] = f"{MODELS_API_URL}?tag={encoded_tag}&nsfw=True"
        models_found: List[Tuple[int, str]] = []
        model_ids_seen: set[int] = set()
        MAX_SEARCH_PAGES = 500 # Define the limit as a constant (or make configurable?)

        self.logger.info(f"Searching for models with tag: '{tag_query}' (Max pages: {MAX_SEARCH_PAGES})")
        client = await self._get_client()
        visited_search_urls: set[str] = set()
        page_count: int = 0
        first_page_validated = False # Flag for Idea #2 implementation (keep for now)

        while url:
             page_count += 1
             
             if page_count >= MAX_SEARCH_PAGES:
                 # Improved logging and console message
                 limit_msg = f"Tag search for '{tag_query}' reached maximum page limit ({MAX_SEARCH_PAGES})."
                 warn_msg = f"{limit_msg} This usually indicates an invalid tag returning too many results. Aborting search for this tag."
                 self.logger.warning(warn_msg)
                 # Print clearer message to console
                 print(f"\nWARNING: {limit_msg}")
                 print(f"         Skipping tag '{tag_query}' - likely invalid.")
                 return [] # Return empty list to signal abortion due to limit

             if url in visited_search_urls: self.logger.warning(f"Model search loop detected: {url}"); break
             visited_search_urls.add(url)

             try:
                 data = await self._search_models_by_tag_page(url, client) # Retries inside helper

                 if data:
                     items = data.get('items', [])
                     metadata = data.get('metadata', {})

                     if not items and page_count == 1:
                         self.logger.warning(f"No models found for tag '{tag_query}' on page 1.")
                         return [] # Exit early, valid case

                     elif items:
                         # --- First Page Validation ---
                         if page_count == 1:
                              tag_found_in_results = False
                              for model in items:
                                   model_tags = {tag.lower() for tag in model.get('tags', []) if isinstance(tag, str)}
                                   if tag_query.lower() in model_tags: tag_found_in_results = True; break
                              if not tag_found_in_results:
                                   msg = f"Tag '{tag_query}' not found in model tags on page 1. Aborting search (likely invalid tag)."
                                   self.logger.warning(msg); 
                                   return [] # Abort
                              else: self.logger.debug(f"Tag '{tag_query}' validated via first page."); first_page_validated = True
                         # --- End First Page Validation ---

                         # Process items
                         new_models = 0
                         for model in items:
                             mid = model.get('id'); mname = model.get('name') or f"Unnamed {mid or '?'}"
                             if isinstance(mid, int) and mid not in model_ids_seen:
                                 models_found.append((mid, mname)); model_ids_seen.add(mid); new_models += 1
                         self.logger.debug(f"Found {new_models} new models for '{tag_query}' on page {page_count}.")

                     # Pagination
                     url = metadata.get('nextPage')
                     if url: await asyncio.sleep(1); 
                     else: break
                 else: # Fetch helper failed
                     self.logger.warning(f"Stopping model search pagination for '{tag_query}' due to fetch failure page {page_count}."); url = None; break

             except Exception as e: # Catch final exceptions from helper
                 self.logger.error(f"Error processing model search page {url} for '{tag_query}': {e}"); url = None; break

        # Return results only if validation passed (implicitly, because we return [] early if it fails)
        self.logger.info(f"Found {len(models_found)} unique models for tag '{tag_query}' after {page_count} pages.")
        return models_found # Return whatever was found before limit/error/end

    # --- Main Execution Method ---
    async def run(self) -> None:
        """
        Main entry point to run the downloader based on initialized configuration.
        Collects identifiers, initializes results, dispatches tasks, and handles finalization.
        """
        self.logger.info(f"Starting Civitai Downloader run (Version {SCRIPT_VERSION})...")
        start_time = time.time()
        if not self.mode: # Check if mode failed to initialize in __init__
             self.logger.critical("No valid mode selected or initialized. Aborting run.")
             return

        self._check_mismatched_arguments() # Log warnings about unused args
        overall_success = True # Flag to track if any critical errors occurred
        tasks = [] # List to hold asyncio tasks for concurrent execution
        identifiers_to_process: List[Tuple[str, str]] = [] # Stores (id_type, identifier_value)
        # Map mode number/type to the base folder name
        option_folder_map: Dict[str, str] = {
            'username': 'Username_Search',
            'model': 'Model_ID_Search',
            'modelVersion': 'Model_Version_ID_Search',
            'tag': 'Model_Tag_Search'
        }

        try:
            # --- 1. Collect Identifiers based on Mode ---
            identifiers_raw: List[str] = []
            id_type: str = "" # The type corresponding to the mode ('username', 'model', etc.)

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
                 # This case should ideally not be reached if _get_mode_choice worked
                 raise ValueError(f"Internal Error: Invalid mode '{self.mode}' reached identifier collection.")

            # Populate identifiers_to_process list and perform basic validation
            if identifiers_raw and id_type:
                 for ident_raw in identifiers_raw:
                      ident_clean = ident_raw.strip()
                      if ident_clean: # Ensure identifier is not just whitespace
                           identifiers_to_process.append((id_type, ident_clean))
                 if not identifiers_to_process:
                      raise ValueError(f"No valid identifiers found after processing input for mode {self.mode}.")
            else:
                 # Handle case where identifier getter returned empty list or id_type missing
                 raise ValueError(f"Failed to retrieve valid identifiers for mode {self.mode}.")
            # --- End Identifier Collection ---

            # Bug #50 enhancement: Print summary of what will be processed
            identifier_names = [ident for _, ident in identifiers_to_process]
            print(f"\n{'='*60}")
            print(f"Processing {len(identifiers_to_process)} {id_type}(s): {', '.join(identifier_names)}")
            print(f"Downloads will run concurrently (in parallel)")
            print(f"{'='*60}\n")

            # --- 2. Initialize Results Structure ---
            for idt, ident in identifiers_to_process:
                 result_key = f"{idt}:{ident}"
                 # Initialize entry for this identifier
                 self.run_results[result_key] = {
                     'success_count': 0, 'skipped_count': 0, 'no_meta_count': 0,
                     'api_items': 0, 'status': 'Pending', 'reason': None,
                     'sub_details': {} # Used only for tag mode's model breakdown
                 }
            # --- End Results Initialization ---


            # --- 3. Process Identifiers (Dispatch based on mode) ---
            option_folder = self._create_option_folder(option_folder_map[id_type]) # Create main mode folder once

            if self.mode == '3': # Tag Search Special Handling
                 disable_prompt_check = self._get_disable_prompt_check()
                 # Use the map to get the base folder name for this mode
                 option_folder = self._create_option_folder(option_folder_map['tag'])
                 self.logger.info(f"--- Starting Tag Search Mode ---")
                 # Get just the tag names from the collected identifiers list for logging
                 tag_names_list = [ident for idt, ident in identifiers_to_process if idt == 'tag']
                 self.logger.info(f"Tags: {tag_names_list}, Prompt Check: {'Disabled' if disable_prompt_check else 'Enabled'}")

                 # Loop through each tag identifier collected earlier
                 for id_type, tag in identifiers_to_process:
                     if id_type != 'tag': continue # Safety check, should only have tags here

                     result_key = f"{id_type}:{tag}" # e.g., "tag:your_tag"
                     self.logger.info(f"--- Processing Tag: {tag} ---")
                     tag_query = tag.replace("_", " ") # Use space for API query
                     sanitized_tag_dir_name = self._clean_path_component(tag) # Use underscore version for dir name
                     tag_dir = os.path.join(option_folder, sanitized_tag_dir_name)
                     os.makedirs(tag_dir, exist_ok=True) # Ensure specific tag dir exists

                     # Initialize list to hold tasks specifically for models under THIS tag
                     tag_tasks = []

                     try:
                         # 1. Search for models matching the tag query
                         # This function now returns List[Tuple[int, str]] or []
                         found_models = await self._search_models_by_tag(tag_query) # Retries handled inside

                         # 2. Handle case where no models are found (or search aborted)
                         if not found_models:
                             self.logger.warning(f"No models found or processed for tag '{tag}'.")
                             # Update status if not already set (e.g., by validation failure in _search..)
                             if self.run_results[result_key]['status'] == 'Pending':                                 
                                 self.run_results[result_key]['status'] = 'Completed (No Models Found or Invalid Tag)' # <-- Use this string
                             continue # Skip to the next tag in the list

                         # 3. Store model mapping for CSV summary
                         async with self.tag_model_mapping_lock:
                             # Ensure list exists for tag, then extend with found models
                             self.tag_model_mapping.setdefault(tag, []).extend(found_models)

                         # 4. Prepare download tasks for found models
                         self.run_results[result_key]['status'] = 'Processing Models'
                         self.logger.info(f"Found {len(found_models)} models for '{tag}'. Queueing downloads...")
                         tag_to_check = tag if not disable_prompt_check else None # Tag name (with _) for prompt check

                         for model_id, model_name in found_models: # Unpack id and name
                             self.logger.debug(f"Queueing task for model ID: {model_id} ('{model_name}') under tag '{tag}'")
                             model_target_dir = os.path.join(tag_dir, f"model_{model_id}")
                             os.makedirs(model_target_dir, exist_ok=True) # Ensure model-specific dir exists
                             url = f"{self.base_url}?modelId={model_id}&nsfw=X" # Image API endpoint for model
                             # Prepare info needed by download/processing steps
                             mode_info = {'tag_to_check': tag_to_check, 'disable_prompt_check': disable_prompt_check, 'current_tag': tag }
                             # Append the task (which calls _run_paginated_download) to the list for this tag
                             tag_tasks.append(self._run_paginated_download(url, model_target_dir, mode_info, parent_result_key=result_key, model_id=model_id))

                         # --- 5. Execute Download Tasks for THIS Tag ---
                         self.logger.info(f"Prepared {len(tag_tasks)} download tasks for tag '{tag}'.") # Log count BEFORE gather
                         if tag_tasks:
                             self.logger.info(f"Executing asyncio.gather for {len(tag_tasks)} tasks (tag '{tag}')...")
                             # Use return_exceptions=True to catch errors within tasks
                             results = await asyncio.gather(*tag_tasks, return_exceptions=True)
                             self.logger.info(f"Finished asyncio.gather for tag '{tag}'.")
                             # Check results for exceptions
                             for i, result in enumerate(results):
                                 if isinstance(result, Exception):
                                     # Log details about the task that failed if possible
                                     # We might need more context passed back from _run_paginated_download
                                     # or infer from the index 'i' which model it might relate to (less reliable).
                                     self.logger.error(f"Task {i} for tag '{tag}' failed within gather: {result}", exc_info=result)
                                     # Mark overall success as False if any task failed internally
                                     overall_success = False
                                     # Optionally update the status of the specific sub-model in run_results here?
                                     # This requires knowing which task corresponds to which model_id.
                         else:
                              self.logger.warning(f"No tasks generated for tag '{tag}', skipping gather.")
                         # --- End Execution Block ---

                         # Mark the overall tag processing as completed ONLY IF no critical error occurred within this try block
                         # Individual model statuses are updated within _run_paginated_download
                         if self.run_results[result_key]['status'] == 'Processing Models': # Avoid overwriting Failed status
                            self.run_results[result_key]['status'] = 'Completed'

                     except Exception as tag_proc_err: # Catch errors during the processing of a single tag (e.g., search failure)
                          self.logger.error(f"Error processing tag {tag}: {tag_proc_err}", exc_info=True)
                          if result_key in self.run_results: # Update status if entry exists
                             self.run_results[result_key]['status'] = 'Failed'
                             # Store the specific error message
                             self.run_results[result_key]['reason'] = f"{type(tag_proc_err).__name__}: {tag_proc_err}"
                          overall_success = False # Mark run as having errors
                 # --- End Tag Processing Loop ---
                 self.logger.info(f"--- Finished Tag Search Mode ---")
            

            else: # Modes 1, 2, 4 (Username, ModelID, ModelVersionID)
                # Get filter tags and prompt check option for Mode 4
                filter_tags_mode4 = None
                disable_prompt_check_mode4 = False
                if self.mode == '4':
                    filter_tags_mode4 = self._get_filter_tags()
                    if filter_tags_mode4:
                        disable_prompt_check_mode4 = self._get_disable_prompt_check()
                        self.logger.info(f"Mode 4 Tag Filtering: {filter_tags_mode4}, Prompt Check: {'Disabled' if disable_prompt_check_mode4 else 'Enabled'}")

                for idt, ident in identifiers_to_process:
                     result_key = f"{idt}:{ident}"; valid, reason = await self._validate_identifier_basic(ident, idt)
                     if not valid: self.run_results[result_key].update({'status':'Failed (Validation)', 'reason': reason}); continue
                     target_dir, url, mode_info = "", "", None
                     url_params = f"&nsfw=X&sort=Newest" if idt == 'username' else "&nsfw=X"
                     if idt == 'username': target_dir = os.path.join(option_folder, self._clean_path_component(ident)); url = f"{self.base_url}?username={ident}{url_params}"
                     elif idt == 'model': target_dir = os.path.join(option_folder, f"model_{ident}"); url = f"{self.base_url}?modelId={ident}{url_params}"
                     elif idt == 'modelVersion':
                         target_dir = os.path.join(option_folder, f"modelVersion_{ident}")
                         url = f"{self.base_url}?modelVersionId={ident}{url_params}"
                         # Apply filter tags for Mode 4 if provided
                         if filter_tags_mode4:
                             # Use first filter tag as the tag to check (combine if multiple)
                             tag_to_check = "_".join(filter_tags_mode4)
                             mode_info = {'tag_to_check': tag_to_check, 'disable_prompt_check': disable_prompt_check_mode4, 'current_tag': None}
                     if target_dir and url: os.makedirs(target_dir, exist_ok=True); tasks.append(self._run_paginated_download(url, target_dir, mode_info, parent_result_key=result_key))
                     else: self.run_results[result_key].update({'status':'Failed', 'reason':'Internal setup error'})
                # --- Execute all collected tasks for modes 1, 2, 4 ---
                if tasks:
                    self.logger.info(f"Executing {len(tasks)} download tasks (Modes 1, 2, 4)...")
                    # Gather tasks for non-tag modes here
                    await asyncio.gather(*tasks) # adding return_exceptions=True here too?
                    self.logger.info("Finished gathering non-tag download tasks.")
                else:
                    self.logger.info("No download tasks were generated for modes 1, 2, or 4.")

        except ValueError as ve: # Catch explicit ValueErrors raised during setup
            self.logger.critical(f"Run setup error: {ve}", exc_info=False) # Log as critical, no need for full traceback
            print(f"\nSETUP ERROR: {ve}")
            overall_success = False
        except Exception as e: # Catch unexpected errors during setup/dispatch
             self.logger.critical(f"Critical error during run setup/dispatch: {e}", exc_info=True)
             print(f"\n--- CRITICAL ERROR ---")
             print(f"An unexpected error stopped the process: {e}")
             print(f"Please check the log file for details: {log_file_path}")
             print(f"----------------------")
             overall_success = False
             # Mark any remaining pending results as failed due to the critical error
             for key, data in self.run_results.items():
                 if data.get('status') == 'Pending':
                     data.update({'status':'Failed', 'reason':f'Run interrupted by critical error: {e}'})
        finally:
             # --- 5. Finalization ---
             self.logger.info("Run finalization steps...")
             final_option_folder = "" # Determine folder for potential CSV summary
             if self.mode and self.mode in option_folder_map: # Check if mode is valid before getting folder
                  final_option_folder = os.path.join(self.output_dir, option_folder_map[self.mode])

             # Close HTTP client if open
             if self._client and not self._client.is_closed:
                 await self._client.aclose(); self.logger.info("HTTP Client closed.")
             # Close DB connection if open
             if self.db_conn:
                 try: self.db_conn.close(); self.logger.info("Database connection closed.")
                 except sqlite3.Error as e: self.logger.error(f"Error closing database connection: {e}")

             # Write summaries (only if Mode 3 was run and folder path is valid)
             if self.mode == '3' and final_option_folder:
                 await self._write_tag_summaries(final_option_folder)

             # Print final statistics report
             self._print_download_statistics()

             # Report specific identifier failures/errors to console
             failed_items = {k:v for k,v in self.run_results.items() if not str(v.get('status','')).startswith('Completed')}
             if failed_items:
                  self.logger.warning("Some identifiers failed processing or completed with errors:")
                  print("\nWarning: Some identifiers had issues:")
                  # Sort failed items for consistent reporting
                  for key in sorted(failed_items.keys()):
                       data = failed_items[key]
                       reason = data.get('reason', 'N/A')
                       self.logger.warning(f"- {key}: Status={data['status']}, Reason={reason}")
                       print(f"- {key}: {data['status']} (Reason: {reason})")

             # Report raw URL failures for debugging (counts only to console)
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

            # This block executes regardless of whether exceptions occurred in the try block.
             self.logger.info("--- Starting Run Finalization ---")
             run_duration = time.time() - start_time
             self.logger.info(f"Total run duration: {run_duration:.2f} seconds")

             # Final status message based on overall success and specific failures
             run_status_msg = 'successfully' if overall_success and not failed_items else 'with errors'
             self.logger.info(f"Run finished {run_status_msg}.")


    # --- Input Gathering Methods ---
    def _get_usernames(self) -> List[str]:
        """Gets usernames from args or interactive prompt."""
        if self.args.username: return [n.strip() for n in self.args.username.split(",") if n.strip()]
        elif self._is_interactive_mode():
             while True: # Loop should already be correct here
                names = input("Enter username(s) (comma-separated): ").strip()
                if names: return [n.strip() for n in names.split(",") if n.strip()]
                else: print("Please enter at least one username.") # Loop continues
        else: self.logger.error("Username (--username) required for Mode 1 in non-interactive mode."); print("Error: Username required."); sys.exit(1)

    def _get_model_ids(self) -> List[str]:
        """Gets model IDs from args or interactive prompt, validating numeric."""
        if self.args.model_id: # Check CLI arg first
            ids_str = str(self.args.model_id) # Ensure it's treated as a string
            ids = [i.strip() for i in ids_str.split(',') if i.strip()]
            if ids and all(i.isdigit() for i in ids): return ids # Return if valid CLI input
            # If CLI input is invalid, log error and exit (non-interactive shouldn't re-prompt)
            self.logger.error(f"Invalid Model ID provided via --model_id: '{self.args.model_id}'. Must be numeric, comma-separated.");
            print(f"Error: Invalid Model ID provided via --model_id: '{self.args.model_id}'. Must be numeric.");
            sys.exit(1)
        elif self._is_interactive_mode(): # Interactive mode
            while True: # Loop until valid input is given
                ids_in = input("Enter model ID(s) (numeric, comma-separated): ").strip()
                ids = [i.strip() for i in ids_in.split(',') if i.strip()]
                if not ids: # Check if input was empty after stripping
                     print("Please enter at least one model ID.")
                     continue # Ask again
                if all(i.isdigit() for i in ids):
                    return ids # Valid input, return the list
                else:
                    # Invalid input, print message and loop continues automatically
                    print("Invalid input. Please enter numeric IDs only, separated by commas.")
        else: # Non-interactive mode, but --model_id was missing
            self.logger.error("Model ID (--model_id) required for Mode 2 in non-interactive mode.");
            print("Error: Model ID required for Mode 2.");
            sys.exit(1) # Exit if required and not provided

    def _get_tags(self) -> List[str]:
        """Gets tags from args or interactive prompt, replacing spaces with underscores."""
        tags_raw = None
        if self.args.tags: tags_raw = self.args.tags
        elif self._is_interactive_mode():
            while True: # Loop should already be correct here
                tags_in = input("Enter tags (comma-separated): ").strip()
                if tags_in: tags_raw = tags_in; break # Exit loop on non-empty input
                else: print("Please enter at least one tag.") # Loop continues
        else: self.logger.error("Tags (--tags) required for Mode 3 in non-interactive mode."); print("Error: Tags required."); sys.exit(1)
        # Process raw tags
        tags = [t.strip().replace(" ", "_") for t in tags_raw.split(',') if t.strip()]
        if not tags: self.logger.error("No valid tags found after processing input."); print("Error: No valid tags provided."); sys.exit(1)
        return tags

    def _get_disable_prompt_check(self) -> bool:
        """Gets the disable_prompt_check option (True/False) from args or interactive prompt."""
        val_map = {'y': True, 'n': False}
        argparse_default = 'n' # The default specified in argparse

        # Check if user explicitly provided a value DIFFERENT from the default via CLI
        cli_value_provided = self.args.disable_prompt_check is not None
        cli_value_is_non_default = cli_value_provided and self.args.disable_prompt_check.lower() != argparse_default

        if cli_value_provided and cli_value_is_non_default:
            # User provided 'y' explicitly
            self.logger.debug(f"Disable prompt check explicitly set via CLI: {self.args.disable_prompt_check}")
            return val_map.get(self.args.disable_prompt_check.lower(), False) # Return True if 'y', False otherwise (should be 'y' here)
        elif self._is_interactive_mode():
            # Only prompt if interactive mode AND the value wasn't explicitly 'y'
            self.logger.debug("Interactive mode detected for disable_prompt_check, prompting user...")
            while True:
                 # Default in prompt text is 'n' (meaning check is enabled by default)
                 resp = input("Disable prompt check? (y/n) [default: n]: ").lower().strip()
                 self.logger.debug(f"User input for disable_prompt_check: '{resp}'")
                 if resp == 'y':
                     self.logger.info("User selected disable prompt check: Yes (y)")
                     return True
                 elif resp == 'n' or resp == '': # Empty input means default 'n'
                     self.logger.info(f"User selected disable prompt check: No (n) (Selected: '{resp}')")
                     return False
                 else:
                     print("Invalid input. Please enter 'y' or 'n'.")
        else:
            # Not interactive mode. Use the value from args (which is the default 'n' unless explicitly set to 'y').
            final_value = val_map.get(self.args.disable_prompt_check.lower(), val_map[argparse_default]) # Default to 'n' -> False if invalid
            self.logger.debug(f"Non-interactive mode, using default/parsed disable_prompt_check value: {final_value}")
            return final_value

    def _get_model_version_ids(self) -> List[str]:
        """Gets model version IDs from args or interactive prompt, validating numeric."""
        if self.args.model_version_id: # Check CLI arg first
            ids_str = str(self.args.model_version_id) # Ensure string
            ids = [i.strip() for i in ids_str.split(',') if i.strip()]
            if ids and all(i.isdigit() for i in ids): return ids # Return if valid CLI input
            # Invalid CLI input
            self.logger.error(f"Invalid Model Version ID provided via --model_version_id: '{self.args.model_version_id}'. Must be numeric.");
            print(f"Error: Invalid Model Version ID provided via --model_version_id: '{self.args.model_version_id}'.");
            sys.exit(1)
        elif self._is_interactive_mode(): # Interactive mode
            while True: # Loop until valid input
                ids_in = input("Enter model version ID(s) (numeric, comma-separated): ").strip()
                ids = [i.strip() for i in ids_in.split(',') if i.strip()]
                if not ids: # Check for empty input
                     print("Please enter at least one model version ID.")
                     continue # Ask again
                if all(i.isdigit() for i in ids):
                    return ids # Valid input
                else:
                    # Invalid input, loop continues
                    print("Invalid input. Please enter numeric IDs only, separated by commas.")
        else: # Non-interactive, arg missing
            self.logger.error("Model Version ID (--model_version_id) required for Mode 4 in non-interactive mode.");
            print("Error: Model Version ID required for Mode 4.");
            sys.exit(1)

    def _get_filter_tags(self) -> Optional[List[str]]:
        """Gets filter tags for Mode 4 from args or interactive prompt. Returns None if not specified."""
        tags_raw = None
        if self.args.filter_tags:
            tags_raw = self.args.filter_tags
        elif self._is_interactive_mode():
            tags_in = input("Enter filter tag(s) (comma-separated, optional, press Enter to skip): ").strip()
            if tags_in:
                tags_raw = tags_in
            else:
                return None  # User chose not to filter
        else:
            return None  # No filter tags in non-interactive mode without --filter_tags

        # Process raw tags
        if tags_raw:
            tags = [t.strip().replace(" ", "_") for t in tags_raw.split(',') if t.strip()]
            if tags:
                return tags
        return None

    # --- Utility and Reporting Methods ---
    def _create_option_folder(self, option_name: str) -> str:
        """Creates mode-specific subfolder within output directory."""
        option_dir = os.path.join(self.output_dir, option_name)
        try: os.makedirs(option_dir, exist_ok=True); self.logger.debug(f"Ensured folder: {option_dir}"); return option_dir
        except OSError as e: self.logger.error(f"Failed create folder '{option_dir}': {e}"); return self.output_dir

        
    def _clean_path_component(self, path_part: str, max_length: Optional[int] = None) -> str:
        """Cleans/shortens a filename or directory name component using simple replacement."""
        self.logger.debug(f"Cleaning path component (max: {max_length}): INPUT='{path_part}'")
        if max_length is None: max_length = self.max_path_length

        # Define the set of characters considered invalid
        invalid_char_set = set('<>:"/\\|?*\t\n\r') | {chr(i) for i in range(32)}

        # Step 1: Decode common URL encodings first
        cleaned = path_part.replace("%20", " ").replace("%2B", "+").replace("%26", "&")

        # Step 2: Replace invalid characters with underscore using a loop
        cleaned_list = []
        for char in cleaned:
            if char in invalid_char_set:
                cleaned_list.append('_')
            else:
                cleaned_list.append(char)
        cleaned = "".join(cleaned_list)
        self.logger.debug(f"After invalid char replace loop: '{cleaned}'")

        # Step 3: Strip leading/trailing problematic chars and consolidate underscores
        cleaned = cleaned.strip('. _') # Strip spaces, dots, and underscores from ends
        cleaned = re.sub(r'_+', '_', cleaned) # Consolidate underscores
        self.logger.debug(f"After strip/consolidate: '{cleaned}'")

        # Step 4: Shorten if necessary
        original_len = len(cleaned)
        if original_len > max_length:
             self.logger.debug(f"Shortening component (len {original_len} > max {max_length})")
             base, dot, ext = cleaned.rpartition('.')
             if dot and len(ext) < 10: # Simple check for a likely extension
                  allowed_base_len = max_length - len(dot) - len(ext)
                  allowed_base_len = max(1, allowed_base_len)
                  new_base = base[:allowed_base_len].strip('_')
                  cleaned = new_base + dot + ext
                  self.logger.debug(f"Shortened preserving extension: '{cleaned}' (Base: '{new_base}', Ext: '{ext}')")
             else: # No extension or very long one, just truncate
                  cleaned = cleaned[:max_length].strip('_')
                  self.logger.debug(f"Shortened via simple truncate: '{cleaned}'")

        # Step 5: Ensure result is not empty
        final_cleaned = cleaned if cleaned else "_"
        self.logger.debug(f"Cleaning result: OUTPUT='{final_cleaned}'")
        return final_cleaned

    def _safe_move(self, src: str, dst: str, max_retries: int = 5, delay: float = 0.5) -> bool:
        """Robust file moving with retries and logging, checking actual outcome."""
        # Ensure source and destination paths are absolute for clarity
        abs_src = os.path.abspath(src)
        abs_dst = os.path.abspath(dst)
        src_basename = os.path.basename(abs_src) # For logging

        self.logger.debug(f"Safe move requested for '{src_basename}':")
        self.logger.debug(f"  Source Abs: {abs_src}")
        self.logger.debug(f"  Dest Abs:   {abs_dst}")

        # Ensure destination directory exists
        dst_dir = os.path.dirname(abs_dst)
        try:
             if not os.path.exists(dst_dir):
                  os.makedirs(dst_dir)
                  self.logger.debug(f"Created destination directory: {dst_dir}")
        except OSError as e:
             self.logger.error(f"Failed to create destination directory '{dst_dir}' for move: {e}")
             return False

        # Check if source exists before starting retries
        if not os.path.exists(abs_src):
            self.logger.warning(f"Source file does not exist at start of safe_move: {abs_src}")
            return False # Cannot move a non-existent file

        move_succeeded_flag = False
        for attempt in range(1, max_retries + 1):
            try:
                # Double-check source existence right before move within loop
                if not os.path.exists(abs_src):
                    self.logger.warning(f"Source file vanished before move attempt {attempt}: {abs_src}")
                    return False # Already gone

                # --- Perform the move ---
                self.logger.debug(f"Attempting shutil.move (Attempt {attempt}/{max_retries})...")
                shutil.move(abs_src, abs_dst)
                # --- Move command completed without raising OS/Permission error ---

                # --- Verify Outcome ---
                # Check if destination exists AFTER the move command
                if os.path.exists(abs_dst):
                    # Check if source STILL exists after the move command
                    if not os.path.exists(abs_src):
                        # This is the expected outcome: destination exists, source is gone.
                        move_succeeded_flag = True
                        self.logger.debug(f"Move successful (dest exists, src gone): '{src_basename}' -> '{os.path.relpath(abs_dst, self.output_dir)}'")
                        return True # Definite success
                    else:
                        # PROBLEM: Dest exists, but source ALSO still exists. shutil.move performed a copy?
                        self.logger.warning(f"shutil.move completed for '{src_basename}', BUT source file still exists at '{abs_src}'. Destination is '{abs_dst}'. Treating as copy, not move.")
                        # Decide how to handle this. For sorting, we WANT the source gone.
                        # Try explicitly removing the source? Risky if it was a cross-fs copy failure.
                        # Let's return False to indicate the 'move' wasn't a true move.
                        move_succeeded_flag = False # Not a true move
                        # Should we delete the source? For sorting, probably yes if dest exists.
                        try:
                             self.logger.warning(f"Attempting to remove source '{abs_src}' after copy-like behavior.")
                             os.remove(abs_src)
                             if not os.path.exists(abs_src):
                                  self.logger.info(f"Successfully removed source '{abs_src}' after copy-like move.")
                                  move_succeeded_flag = True # Now it's effectively moved
                                  return True
                             else:
                                  self.logger.error(f"Failed to remove source '{abs_src}' after copy-like move.")
                                  return False # Failed to clean up
                        except OSError as rm_err:
                             self.logger.error(f"Error removing source '{abs_src}' after copy-like move: {rm_err}")
                             return False # Failed to clean up

                else:
                    # PROBLEM: shutil.move didn't error, but destination doesn't exist? Very weird.
                    self.logger.error(f"shutil.move completed for '{src_basename}' without error, BUT destination file '{abs_dst}' does NOT exist. Source exists: {os.path.exists(abs_src)}")
                    move_succeeded_flag = False
                    return False # Treat as failure

            except (PermissionError, OSError) as e:
                # Move failed with an expected error
                if attempt < max_retries:
                    self.logger.debug(f"Move attempt {attempt}/{max_retries} failed for '{src_basename}' ({e}), retrying in {delay * attempt:.1f}s...")
                    time.sleep(delay * attempt)
                else: # Final attempt failed
                    self.logger.error(f"Failed to move '{src_basename}' after {max_retries} attempts: {e}")
                    return False # Indicate final failure
            except Exception as e_unexp: # Catch any other unexpected error during move/checks
                 self.logger.error(f"Unexpected error during safe_move for '{src_basename}' attempt {attempt}: {e_unexp}", exc_info=True)
                 if attempt < max_retries:
                     time.sleep(delay * attempt) # Still retry on unexpected error? Maybe.
                 else:
                     return False # Fail after retries on unexpected error too.


        # If loop finishes without returning True (shouldn't happen normally)
        self.logger.error(f"safe_move loop finished unexpectedly for '{src_basename}'. Final status uncertain.")
        return move_succeeded_flag # Return status based on checks inside loop

        
    def _print_download_statistics(self) -> None:
        """Prints the final download statistics summary, including per-identifier results and warnings."""
        print("\n--- Download Statistics Summary ---")

        if not self.run_results: # Handle case where no identifiers were processed at all
             print("No identifiers processed in this run.")
             print("---------------------------------\n")
             self.logger.info("Run Stats Aggregated: Success=0, Skipped=0, NoMeta=0, API Items=0")
             return

        # --- Stage 1: Calculate Final Aggregated Counts Per Identifier ---
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
            # Calculate final aggregated counts for this identifier
            agg_api = data.get('api_items', 0)
            agg_dl = data.get('success_count', 0)
            agg_skip = data.get('skipped_count', 0)
            agg_nometa = data.get('no_meta_count', 0)

            # Add sub-details if it's a tag
            if key.startswith('tag:') and 'sub_details' in data:
                 agg_api = sum(sd.get('api_items', 0) for sd in data['sub_details'].values())
                 agg_dl = sum(sd.get('success_count', 0) for sd in data['sub_details'].values())
                 agg_skip = sum(sd.get('skipped_count', 0) for sd in data['sub_details'].values())
                 agg_nometa = sum(sd.get('no_meta_count', 0) for sd in data['sub_details'].values())

            identifier_final_counts[key] = {
                'api': agg_api, 'dl': agg_dl, 'skip': agg_skip, 'nometa': agg_nometa
            }

            # Check if this identifier resulted in no processing using aggregated counts
            current_status = data.get('status', 'Unknown')
            is_no_process_status = current_status in no_process_statuses
            all_counts_zero = (agg_api == 0 and agg_dl == 0 and agg_skip == 0 and agg_nometa == 0)

            # --- DETAILED DEBUG LOG ---
            self.logger.debug(f"Checking no-process for '{key}':")
            self.logger.debug(f"  Status = '{current_status}'")
            self.logger.debug(f"  Is Status in no_process_set? {is_no_process_status} (Set: {no_process_statuses})")
            self.logger.debug(f"  Are all counts zero? {all_counts_zero} (A={agg_api},D={agg_dl},S={agg_skip},N={agg_nometa})")
            # --- END DETAILED DEBUG LOG ---

            if is_no_process_status and all_counts_zero:
                 self.logger.debug(f"  -> Adding '{key}' to no-process list.")
                 no_models_processed_identifiers.append(key)
        # --- End Calculation ---

        # --- Stage 2: Calculate Overall Totals ---
        total_api_items = sum(counts['api'] for counts in identifier_final_counts.values())
        total_downloaded = sum(counts['dl'] for counts in identifier_final_counts.values())
        total_skipped = sum(counts['skip'] for counts in identifier_final_counts.values())
        total_no_meta = sum(counts['nometa'] for counts in identifier_final_counts.values())

        # --- Print Overall Aggregates ---
        print(f"Total API items processed (approx): {total_api_items}")
        print(f"Total successful downloads this run: {total_downloaded}")
        print(f"Total images without metadata: {total_no_meta}")
        print(f"Total skipped/failed items: {total_skipped}")

        # Print aggregated skip/fail reasons
        if self.skipped_reasons_summary:
            print("\nReasons for skipping/failing items across run:")
            sorted_reasons = sorted(self.skipped_reasons_summary.items(), key=lambda item: item[1], reverse=True)
            for reason, count in sorted_reasons: print(f"- {reason}: {count} times")

        # --- Stage 3: Print Per-Identifier Breakdown (using calculated counts) ---
        print("\n--- Results per Identifier ---")
        sorted_keys = sorted(self.run_results.keys())
        for key in sorted_keys:
             data = self.run_results[key]
             counts = identifier_final_counts[key] # Get pre-calculated aggregates
             try: id_type, identifier = key.split(":", 1)
             except ValueError: identifier, id_type = key, "Unknown"

             print(f"Identifier: {identifier} (Type: {id_type})")
             print(f"  Status: {GREEN if data.get('status', 'Unknown') == 'Completed' else RED if data.get('status', 'Unknown') == 'Failed' else RESET}{data.get('status', 'Unknown')}{RESET}")
             if data.get('reason'): print(f"  Reason: {data['reason']}")
             # Print the final aggregated counts for this identifier
             print(f"  API Items: {counts['api']}")
             print(f"  Downloaded: {counts['dl']}")
             print(f"  Skipped/Failed: {counts['skip']}")
             print(f"  No Metadata: {counts['nometa']}")
             print("-" * 10)
        # --- End Per-Identifier Breakdown ---

        # --- Stage 4: Print Summary Warning for Unprocessed Identifiers ---
        self.logger.debug(f"Final list of no-process identifiers: {no_models_processed_identifiers}")
        if no_models_processed_identifiers:  
             print("\nNOTE: The following identifiers resulted in zero models/images being processed:")
             self.logger.debug("NOTE: The following identifiers resulted in zero models/images being processed:")
             for key in sorted(no_models_processed_identifiers):
                  status = self.run_results[key].get('status', 'Unknown')
                  reason = self.run_results[key].get('reason')
                  reason_str = f" (Reason: {reason})" if reason else ""
                  msg = f"- {key} (Status: {status}{reason_str})"
                  print(msg) 
                  self.logger.debug(msg) # Log detail as warning too
        # --- End Summary Warning ---

        print("---------------------------------\n")
        self.logger.info(f"Run Stats Aggregated: Success={total_downloaded}, Skipped={total_skipped}, NoMeta={total_no_meta}, API Items={total_api_items}")

    def _check_mismatched_arguments(self) -> None:
        """Logs warnings if CLI arguments conflict with the selected mode."""
        if not self.mode or self.mode not in ['1', '2', '3', '4']: return
        mode = int(self.mode); relevant_args = []; unused_args = []
        if mode == 1: relevant_args = ['username']
        elif mode == 2: relevant_args = ['model_id']
        elif mode == 3: relevant_args = ['tags', 'disable_prompt_check']
        elif mode == 4: relevant_args = ['model_version_id', 'filter_tags', 'disable_prompt_check']
        all_mode_args = ['username', 'model_id', 'model_version_id', 'tags', 'filter_tags', 'disable_prompt_check']
        for argn in all_mode_args:
            argval = getattr(self.args, argn, None)
            is_default = False
            if argn == 'disable_prompt_check': is_default = (argval == 'n') # Check default specifically for disable_prompt_check
            if argval is not None and not is_default and argn not in relevant_args: unused_args.append(f"--{argn.replace('_', '-')}")
        if unused_args: msg = f"Warning: Arguments potentially unused in mode {mode}: {', '.join(unused_args)}"; self.logger.warning(msg); print(msg)

     
    # ============================
    # --- Sorting Logic ---
    # ============================
    async def _sort_images_by_model_name(self, base_dir: str) -> None:
        """
        Sorts downloaded images and metadata files within a directory (base_dir)
        into subfolders based on the 'Model' field found in the metadata files.
        Handles files with no or invalid metadata separately.
        Args:
            base_dir: The directory containing the mixed downloads for one identifier.
        """
        self.logger.info(f"Starting sort process in: {base_dir}")
        if not os.path.isdir(base_dir):
             self.logger.warning(f"Sort directory does not exist: {base_dir}")
             return

        no_meta_dir = os.path.join(base_dir, 'no_metadata')
        invalid_meta_dir = os.path.join(base_dir, 'invalid_metadata')
        # Set to keep track of filenames (not full paths) that have been successfully moved
        processed_filenames: set[str] = set()

        try: # List files safely
             all_entries = list(os.scandir(base_dir))
             all_files_in_basedir = {entry.name for entry in all_entries if entry.is_file()} # Use a set for faster lookups
        except OSError as e:
             self.logger.error(f"Cannot list files in sort directory {base_dir}: {e}")
             return

        meta_files = [f for f in all_files_in_basedir if f.endswith(('_meta.txt', '_no_meta.txt'))]

        if not meta_files:
            self.logger.info(f"No metadata files found to process in {base_dir}")
            return # No sorting needed if no meta files

        # Ensure destination directories exist
        os.makedirs(no_meta_dir, exist_ok=True)
        os.makedirs(invalid_meta_dir, exist_ok=True)
        self.logger.debug(f"Found {len(meta_files)} meta files to process in {base_dir}")

        for meta_file in meta_files:
            # Check if this meta file itself was already moved (e.g., as part of processing a previous file pair)
            if meta_file in processed_filenames:
                self.logger.debug(f"Skipping already processed file: {meta_file}")
                continue

            meta_path = os.path.join(base_dir, meta_file)
            base_name: Optional[str] = None

            # Extract base name correctly using rsplit
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
                # 1. Determine Target Directory based on meta file type and content
                if meta_file.endswith('_no_meta.txt'):
                    target_dir_for_pair = no_meta_dir
                    self.logger.debug(f"Target for '{meta_file}' is no_metadata folder.")
                elif meta_file.endswith('_meta.txt'):
                    model_name = None
                    try:
                        # Read meta file content (sync read okay here)
                        with open(meta_path, 'r', encoding='utf-8') as f: content = f.read()
                        match = re.search(r"^Model:\s*(.+?)\s*$", content, re.IGNORECASE | re.MULTILINE)
                        if match: model_name = match.group(1).strip()
                        self.logger.debug(f"Read '{meta_file}'. Found model: {model_name}")
                    except FileNotFoundError: # Handle case where meta file vanished between listing and reading
                         self.logger.warning(f"Meta file vanished before reading: {meta_path}. Skipping.")
                         continue
                    except Exception as read_e:
                        self.logger.error(f"Failed to read metadata file {meta_path}: {read_e}")
                        model_name = None # Treat as invalid if read fails

                    # Assign target based on model name found
                    if model_name and model_name != 'unknown':
                         # Clean and shorten the model name for use as a directory name
                         model_subdir_name = self._clean_path_component(model_name, max_length=60)
                         target_dir_for_pair = os.path.join(base_dir, model_subdir_name)
                         os.makedirs(target_dir_for_pair, exist_ok=True) # Ensure model subdir exists
                         self.logger.debug(f"Target for '{meta_file}' is model folder: '{model_subdir_name}'")
                    else:
                         target_dir_for_pair = invalid_meta_dir
                         self.logger.debug(f"Target for '{meta_file}' is invalid_metadata folder.")

                if not target_dir_for_pair: # Safety check
                    self.logger.error(f"Logic error: Could not determine target directory for {meta_file}. Skipping.")
                    continue

                # 2. Move Meta File
                target_meta_path = os.path.join(target_dir_for_pair, meta_file)
                self.logger.debug(f"Attempting move: {meta_path} -> {target_meta_path}")
                if self._safe_move(meta_path, target_meta_path):
                    processed_filenames.add(meta_file) # Mark meta file as successfully processed
                    self.logger.debug(f"Meta move successful for {meta_file}")

                    # 3. Move Corresponding Image File (only if meta moved successfully)
                    possible_image_found = False
                    image_moved = False
                    image_extensions = ['.jpeg', '.png', '.webp', '.mp4', '.webm']
                    for ext in image_extensions:
                        image_file = base_name + ext # Use the correctly extracted base name
                        # Check if image exists in original dir AND hasn't been moved yet
                        if image_file in all_files_in_basedir and image_file not in processed_filenames:
                            possible_image_found = True
                            image_path = os.path.join(base_dir, image_file)
                            target_image_path = os.path.join(target_dir_for_pair, image_file)
                            self.logger.debug(f"Found potential image '{image_file}'. Attempting move to: {target_dir_for_pair}")
                            if self._safe_move(image_path, target_image_path):
                                processed_filenames.add(image_file) # Mark image as processed
                                image_moved = True
                                self.logger.debug(f"Image move successful for {image_file}")
                            else:
                                self.logger.warning(f"Failed to move image file {image_file} for meta {meta_file}")
                            break # Found the corresponding image, stop checking extensions
                    if not possible_image_found:
                         self.logger.debug(f"No corresponding image found in {base_dir} for base name '{base_name}' (checked: {image_extensions})")
                    elif not image_moved:
                         self.logger.warning(f"Image file found for {meta_file} but failed to move.")

                else: # Meta move failed
                    self.logger.warning(f"Failed to move meta file {meta_file}, skipping associated image move.")
                    # Do not add meta_file to processed_filenames if move failed

            except Exception as e: # Catch errors during the processing of a single meta file
                 self.logger.error(f"Unexpected error during sort processing for {meta_file}: {e}", exc_info=True)
                 # Continue to the next meta file

        
        # =====================
        # --- Orphan Check ---
        # =====================
        self.logger.debug("Checking for orphaned files after sorting...")
        try:
             # Rescan base directory files after potential moves
             current_files_after_sort = {f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))}
             # Find files remaining in base dir that weren't successfully moved and aren't CSVs
             remaining_files = current_files_after_sort - processed_filenames - {f for f in current_files_after_sort if f.lower().endswith('.csv')}

             orphaned_images = []
             image_extensions_set = {'.jpeg', '.png', '.webp', '.mp4', '.webm'}
             for fname in remaining_files:
                 base_name_orphan, ext_orphan = os.path.splitext(fname)
                 # Check if it looks like one of our images/videos and has a numeric ID name
                 if ext_orphan.lower() in image_extensions_set and base_name_orphan.isdigit():
                      # Check if a corresponding meta file exists anywhere plausible
                      meta_exists = False
                      for meta_suffix in ['_meta.txt', '_no_meta.txt']:
                           possible_meta = base_name_orphan + meta_suffix
                           locations_to_check = [base_dir, no_meta_dir, invalid_meta_dir]
                           try: # Add model subdirs if they exist
                               potential_model_dirs = [d.path for d in os.scandir(base_dir) if d.is_dir() and d.name not in ['no_metadata', 'invalid_metadata']]
                               locations_to_check.extend(potential_model_dirs)
                           except OSError: pass # Ignore errors listing dirs (e.g., permission denied)

                           for loc in locations_to_check:
                               try: # Check existence safely
                                   if os.path.exists(os.path.join(loc, possible_meta)):
                                       meta_exists = True; break
                               except OSError: continue # Ignore errors checking specific paths
                           if meta_exists: break # Found meta for this orphan base name

                      if not meta_exists: # If no meta found anywhere after checking all locations
                           orphaned_images.append(fname)

             if orphaned_images:
                  # Only log warning about potential orphans
                  self.logger.warning(f"Found potential orphaned image files in {base_dir}: {orphaned_images}")

        except OSError as e:
            self.logger.warning(f"Could not perform orphan check in {base_dir}: {e}")

        self.logger.info(f"Finished sort process in: {base_dir}")

    
    # ============================
    # --- Tag Summary CSV ---
    # ============================
    async def _write_tag_summaries(self, option_folder: str) -> None:
        """Writes CSV summaries for tags using SQLite data."""
        self.logger.info("Generating tag summary CSV files...");
        if not self.db_conn: self.logger.error("DB connection unavailable."); return
        async with self.tag_model_mapping_lock: tags_processed = list(self.tag_model_mapping.keys())
        if not tags_processed: self.logger.info("No tags processed, skipping CSV."); return
        for tag in tags_processed:
             sanitized_tag = self._clean_path_component(tag); tag_dir = os.path.join(option_folder, sanitized_tag)
             try: os.makedirs(tag_dir, exist_ok=True)
             except OSError as e: self.logger.error(f"Failed create CSV dir '{tag_dir}': {e}"); continue
             csv_path = os.path.join(tag_dir, f"summary_{sanitized_tag}_{datetime.now().strftime('%Y%m%d')}.csv"); csv_data = []
             try: # Query DB
                 async with self.tracking_lock: # Read lock needed? Maybe not.
                    cursor = self.db_conn.cursor()
                    cursor.execute('''SELECT T1.path, T1.url, T1.checkpoint_name, T2_All.tag FROM tracked_images T1 JOIN image_tags T_Current ON T1.image_key = T_Current.image_key AND T_Current.tag = ? LEFT JOIN image_tags T2_All ON T1.image_key = T2_All.image_key AND T2_All.tag != ?''', (tag, tag))
                    rows = cursor.fetchall()
                 processed = {} # path -> {data}
                 for path, url, cp_name, other_tag in rows:
                     if path not in processed: processed[path] = {"cp": cp_name or "Unknown", "url": url or "N/A", "others": set()}
                     if other_tag: processed[path]["others"].add(other_tag)
                 # Build CSV rows
                 for img_path, data in processed.items():
                     rel_path = img_path; # Default
                     if img_path and os.path.exists(tag_dir) and os.path.exists(img_path):
                          try: rel_path = os.path.relpath(img_path, tag_dir)
                          except ValueError: pass # Keep abs path if different drives
                     others_list = sorted(list(data["others"]))
                     if others_list: csv_data.extend([[tag, ot, data["cp"], rel_path, data["url"]] for ot in others_list])
                     else: csv_data.append([tag, "", data["cp"], rel_path, data["url"]])
             except sqlite3.Error as e: self.logger.error(f"DB query error for tag '{tag}': {e}"); continue
             if not csv_data: self.logger.info(f"No images found for tag '{tag}' summary."); continue
             # Write CSV
             try:
                 with open(csv_path, "w", newline="", encoding='utf-8') as f:
                     w = csv.writer(f); w.writerow(["Current Tag", "Previously Downloaded Tag", "Checkpoint Name", "Relative Image Path", "Download URL"]); w.writerows(csv_data)
                 self.logger.info(f"Wrote summary CSV: {csv_path}")
             except Exception as e: self.logger.error(f"Failed write CSV for '{tag}': {e}", exc_info=True)

# ===================
# --- Entry Point ---
# ===================
if __name__ == "__main__":
    cli_args = parse_arguments()
    # Set global retry count from args before creating downloader
    CURRENT_RETRY_COUNT = cli_args.retries
    is_cli = cli_args.mode is not None
    print(f"--- Civitai Downloader v{SCRIPT_VERSION} ---")
    print(f"Running in {'command-line' if is_cli else 'interactive'} mode.")
    logger.info(f"Running in {'command-line' if is_cli else 'interactive'} mode.")
    try:
        downloader = CivitaiDownloader(cli_args)
        asyncio.run(downloader.run())
    except SystemExit: logger.warning("Exiting (SystemExit).") # More specific exit log
    except KeyboardInterrupt: logger.warning("Process interrupted by user (Ctrl+C)."); print("\nProcess interrupted.")
    except Exception as main_err:
        logger.critical(f"Unhandled exception at main level: {main_err}", exc_info=True)
        print(f"\n--- UNHANDLED CRITICAL ERROR ---\nError: {main_err}\nCheck log: {log_file_path}\n----------------------")
    logger.info("Script finished.")
    print("\nDownload process complete.")
    
