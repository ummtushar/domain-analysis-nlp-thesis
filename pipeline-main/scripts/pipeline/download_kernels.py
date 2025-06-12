import csv
from utils.kaggle_api import get_kaggle_api_instance
import logging
from utils.config_handler import read_property, write_property
from utils.logger import record_error

api = get_kaggle_api_instance()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _download_kernel(ref: str, out_path: str) -> bool:
    """
    Downloads a kernel using the provided reference and saves it to the specified output path.

    Args:
        ref (str): The reference identifier for the kernel to be downloaded.
        out_path (str): The file path where the downloaded kernel will be saved.

    Returns:
        bool: True if the kernel was successfully downloaded, False otherwise.

    Logs:
        Logs an info message if the kernel is successfully downloaded.
        Logs an error message if the download fails and records the error.
    """
    try:
        api.kernels_pull(ref, out_path, metadata=True)
        logger.info(f'Successfully downloaded kernel: {ref}')
        return True
    except Exception as e:
        logger.error(f'Failed to download kernel: {ref}, error: {e}')
        record_error(ref, e)
        return False

def download_kernels(csv_path: str, out_dir: str, config_path: str) -> bool: 
    """
    Downloads kernels listed in a CSV file to a specified output directory and updates the current kernel property.

    Args:
        csv_path (str): The file path to the CSV file containing kernel references.
        out_dir (str): The directory where the downloaded kernels will be saved.
        config_path (str): The file path to the configuration file.

    Returns:
        bool: True if all kernels are downloaded successfully, False otherwise.
    """
    section = 'DEFAULT'
    _property = 'current_kernel'
    current_kernel = int(read_property(section, _property, config_path))

    with open(csv_path) as file:
        reader = csv.DictReader(file)
        for kernel in reader:
            _id = int(kernel['id'])
            print(f'ID: {_id}')
            if _id < current_kernel:
                continue

            ref = kernel['ref']
            downloaded = _download_kernel(ref, f'{out_dir}/{ref}')
            if not downloaded:
                write_property(section, _property, _id, config_path)
                return False

    return True
        