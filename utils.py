import os
import multiprocessing
from pathlib import Path
import requests  # Use the more robust 'requests' library
from functools import partial
from tqdm import tqdm

def download_image_worker(image_link, savefolder):
    """
    Safely downloads a single image using the requests library with a timeout.
    This is the function that will be run by each parallel process.
    """
    if not isinstance(image_link, str) or not image_link.startswith('http'):
        return  # Skip if the link is not a valid string or URL

    try:
        # Create a clean filename from the URL, removing query parameters
        filename = Path(image_link).name.split('?')[0]
        if not filename:
            return # Skip if no valid filename can be made

        image_save_path = os.path.join(savefolder, filename)

        # Only download if the file doesn't already exist to save time
        if not os.path.exists(image_save_path):
            # Make the web request with a 10-second timeout
            with requests.get(image_link, stream=True, timeout=10) as r:
                r.raise_for_status()  # This will raise an error for bad responses (404, 500, etc.)
                with open(image_save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    except Exception as e:
        # Catch any error (timeout, connection error, bad status) and print a warning
        # This prevents a single bad link from crashing the entire script
        print(f'\nWarning: Could not download {image_link}. Reason: {e}')

def download_images(image_links, download_folder):
    """
    Downloads a list of images in parallel using a safe number of processes.
    """
    os.makedirs(download_folder, exist_ok=True)

    # Use a safe number of processes to avoid overwhelming the system
    num_processes = min(16, os.cpu_count() * 2)

    # Create a partial function to pass the save folder to the worker
    worker = partial(download_image_worker, savefolder=download_folder)

    # Create a process pool to execute the downloads in parallel
    with multiprocessing.Pool(num_processes) as pool:
        # This approach is more robust for handling the pool's lifecycle.
        # We iterate through the results immediately, which helps ensure processes close properly.
        for _ in tqdm(pool.imap_unordered(worker, image_links), total=len(image_links), desc="Downloading Images"):
            pass

