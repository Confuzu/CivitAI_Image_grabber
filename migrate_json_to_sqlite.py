# migrate_json_to_sqlite.py
import sqlite3
import json
import os
import sys
from datetime import datetime
import logging
from typing import Optional, List, Tuple

# --- Configuration ---
OLD_JSON_FILENAME = "downloaded_images.json"
NEW_DB_FILENAME = "tracking_database.sqlite"
BATCH_SIZE = 500 # Process records in batches

LOG_FILENAME_TEMPLATE: str = "migration_tool_log_{version}.txt"
SCRIPT_VERSION: str = "0.1_migration" # Current script version for logging

# --- Logging Setup (Global) ---
script_dir: str = os.path.dirname(os.path.abspath(__file__))
log_file_path: str = os.path.join(script_dir, LOG_FILENAME_TEMPLATE.format(version=SCRIPT_VERSION))
logger: logging.Logger = logging.getLogger('migration_tool')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # File Handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
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

# --- Database Functions ---
def initialize_db(db_path: str) -> Optional[sqlite3.Connection]:
    """Connects to DB and ensures schema exists."""
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        logger.info(f"Connected to database: {db_path}")
        cursor = conn.cursor()

        # Enable Foreign Keys (optional but recommended)
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Create tracked_images table (NO tags column)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracked_images (
                image_key TEXT PRIMARY KEY, -- Format: "{image_id}_{quality}"
                image_id TEXT NOT NULL,
                quality TEXT NOT NULL,
                path TEXT NOT NULL,
                download_date TEXT NOT NULL,
                url TEXT,
                checkpoint_name TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracked_images_key ON tracked_images (image_key)') # Renamed index

        # Create image_tags table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_tags (
                image_key TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (image_key, tag),
                FOREIGN KEY(image_key) REFERENCES tracked_images(image_key) ON DELETE CASCADE
            )
        ''')
        # Index for querying by tag
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags (tag)')
        # Index for joining (covered by primary key, but explicit index might sometimes help)
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_tags_key ON image_tags (image_key)')

        conn.commit()
        logger.info("Database schema verified/created.")
        return conn
    except sqlite3.Error as e:
        logger.critical(f"Database initialization error: {e}", exc_info=True)
        if conn:
            try: conn.close()
            except sqlite3.Error: pass
        return None

def insert_image_batch(cursor: sqlite3.Cursor, batch: List[Tuple]):
    """Inserts a batch of image data."""
    # Using INSERT OR IGNORE to handle potential duplicates if migration is re-run
    cursor.executemany('''
        INSERT OR IGNORE INTO tracked_images
        (image_key, image_id, quality, path, download_date, url, checkpoint_name)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', batch)
    return cursor.rowcount

def insert_tag_batch(cursor: sqlite3.Cursor, batch: List[Tuple]):
    """Inserts a batch of tag data."""
    # Using INSERT OR IGNORE for tags as well
    cursor.executemany('''
        INSERT OR IGNORE INTO image_tags (image_key, tag) VALUES (?, ?)
    ''', batch)
    # We don't easily get a count of *new* tags inserted here with IGNORE

# --- Main Migration Logic ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, OLD_JSON_FILENAME)
    db_path = os.path.join(script_dir, NEW_DB_FILENAME)
    conn: Optional[sqlite3.Connection] = None # Initialize conn
    migration_successful = False # Flag to track success

    logger.info("--- Civitai Downloader JSON to SQLite Migration Tool ---")

    # --- Pre-checks ---
    if not os.path.exists(json_path):
        logger.error(f"Old JSON file '{json_path}' not found. Nothing to migrate.")
        sys.exit(1)

    if os.path.exists(db_path):
        confirm = input(f"Database file '{db_path}' already exists. Continue? (Data will be added/updated) (y/N): ").strip().lower()
        if confirm != 'y':
            logger.info("Migration cancelled by user.")
            sys.exit(0)

    try:
        # --- Load JSON ---
        logger.info(f"Loading JSON data from '{json_path}'...")
        try:
            with open(json_path, "r", encoding='utf-8') as f:
                downloaded_images_json = json.load(f)
            if not isinstance(downloaded_images_json, dict):
                logger.error("JSON file does not contain a valid dictionary.")
                sys.exit(1)
            logger.info(f"Loaded {len(downloaded_images_json)} entries from JSON.")
        except Exception as e:
            logger.critical(f"Failed to load or parse JSON file: {e}", exc_info=True)
            sys.exit(1)

        # --- Initialize Database ---
        conn = initialize_db(db_path)
        if not conn:
            sys.exit(1) # Initialization failed

        # --- Migrate Data ---
        logger.info("Starting data migration...")
        cursor = conn.cursor()
        total_processed = 0
        total_images_inserted = 0
        total_tags_processed = 0
        skip_count = 0
        image_batch: List[Tuple] = []
        tag_batch: List[Tuple] = []

        for image_key, info in downloaded_images_json.items():
            total_processed += 1
            try: # Process each item individually within the main try
                if not image_key or '_' not in image_key or not isinstance(info, dict): raise ValueError("Invalid key or info structure")
                image_id_str, quality = image_key.split('_', 1)
                path = info.get("path"); download_date = info.get("download_date")
                if not path or not download_date: raise ValueError("Missing path or date")

                url = info.get("url"); checkpoint_name = info.get("checkpoint_name")
                image_batch.append((image_key, image_id_str, quality, path, download_date, url, checkpoint_name))

                tags = info.get("tags", [])
                if isinstance(tags, list):
                    unique_tags = set(t for t in tags if t)
                    for tag in unique_tags: tag_batch.append((image_key, tag)); total_tags_processed += 1
                elif tags: logger.warning(f"Invalid 'tags' format for {image_key}, skipping tags.")

                # Insert batches
                if len(image_batch) >= BATCH_SIZE:
                    inserted = insert_image_batch(cursor, image_batch); total_images_inserted += inserted
                    conn.commit(); image_batch = []; logger.info(f"Processed {total_processed} JSON entries...")
                if len(tag_batch) >= BATCH_SIZE * 5: # Allow tag batch to grow larger
                    insert_tag_batch(cursor, tag_batch); conn.commit(); tag_batch = []

            except Exception as item_err:
                 logger.error(f"Error processing item {image_key}: {item_err}")
                 skip_count += 1
                 # Optionally rollback here if needed, but IGNORE should prevent bad data insert
                 # conn.rollback()

        # Insert remaining items
        if image_batch: inserted = insert_image_batch(cursor, image_batch); total_images_inserted += inserted
        if tag_batch: insert_tag_batch(cursor, tag_batch)
        conn.commit() # Final commit for any remaining items

        # If we reach here without critical errors, consider it successful
        migration_successful = True
        logger.info("--- Migration Summary ---")
        logger.info(f"Total JSON entries processed: {total_processed}")
        # Correct ignored count calculation
        ignored_count = total_processed - total_images_inserted - skip_count
        logger.info(f"Image records newly inserted/ignored: {total_images_inserted}/{ignored_count}")
        logger.info(f"Tag relationships processed: {total_tags_processed}")
        logger.info(f"Entries skipped due to errors: {skip_count}")

    except sqlite3.Error as db_err:
        logger.critical(f"Database error during migration: {db_err}", exc_info=True)
        migration_successful = False # Mark as failed
    except Exception as e:
        logger.critical(f"An unexpected error occurred during migration: {e}", exc_info=True)
        migration_successful = False # Mark as failed
    finally:
        # --- Ensure DB connection is closed ---
        if conn:
            try: conn.close()
            except sqlite3.Error as close_err: logger.error(f"Error closing database: {close_err}")
            logger.info("Database connection closed.")

    # --- Perform Rename *AFTER* try/finally block ---
    if migration_successful:
        confirm_rename = input(f"Migration finished successfully. Rename '{OLD_JSON_FILENAME}' to prevent re-migration? (y/N): ").strip().lower()
        if confirm_rename == 'y':
            try:
                # Use absolute paths calculated at the start
                abs_script_dir = os.path.dirname(os.path.abspath(__file__))
                abs_json_path = os.path.join(abs_script_dir, OLD_JSON_FILENAME)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                new_filename = f"{OLD_JSON_FILENAME}.migrated_to_sqlite_{timestamp}"
                abs_new_file_path = os.path.join(abs_script_dir, new_filename)

                logger.info(f"Attempting rename: '{abs_json_path}' -> '{abs_new_file_path}'")

                if os.path.exists(abs_json_path):
                     os.rename(abs_json_path, abs_new_file_path)
                     logger.info(f"Successfully Renamed '{OLD_JSON_FILENAME}'.")
                else:
                     # This case should ideally not happen if migration_successful is True, but handle it.
                     logger.warning(f"Original JSON file '{abs_json_path}' not found. Cannot rename.")

            except OSError as ren_err:
                logger.error(f"Could not rename old JSON file '{abs_json_path}': {ren_err}", exc_info=True)
            except Exception as e:
                 logger.error(f"Unexpected error during rename phase: {e}", exc_info=True)
        else:
            logger.info("Old JSON file not renamed.")
    else:
         logger.warning("Migration did not complete successfully. Old JSON file will NOT be renamed.")

if __name__ == "__main__":
    main()