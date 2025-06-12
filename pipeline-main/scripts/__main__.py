import logging
from pipeline.download_kernels import download_kernels
from pipeline.fetch_kernels_metadata import create_csv, retrieve_kernel_metadata
import time
from utils.config_handler import read_property, write_property
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function for the Kaggle Kernel Management Tool.

    This function parses command-line arguments to determine the task to perform,
    either 'metadata' or 'download'. Based on the task, it either retrieves kernel
    metadata and creates a CSV file, or downloads kernels and updates the configuration
    file.

    Command-line arguments:
    task (str): Task to perform, either 'metadata' or 'download'.

    The function performs the following tasks:
    - If the task is 'metadata', it retrieves kernel metadata and creates a CSV file.
    - If the task is 'download', it downloads kernels, updates the configuration file,
      and logs the progress, with a sleep interval of 60 seconds between downloads.

    Paths used:
    - csv_path: Path to the CSV file where kernel metadata is stored.
    - out_dir: Directory where downloaded kernels are stored.
    - config_path: Path to the configuration file.
    """
    parser = argparse.ArgumentParser(description='Kaggle Kernel Management Tool')
    parser.add_argument('task', choices=['metadata', 'download'], help='Task to perform: metadata or download')
    args = parser.parse_args()

    csv_path = input('Enter the path for the CSV file (default: jupyter-nbs/data/nbs.csv): ') or 'jupyter-nbs/data/nbs.csv'
    out_dir = input('Enter the output directory for downloaded kernels (default: jupyter-nbs/data/nbs/): ') or 'jupyter-nbs/data/nbs/'
    config_path = input('Enter the path for the configuration file (default: jupyter-nbs/data/config.properties): ') or 'jupyter-nbs/data/config.properties'

    if args.task == 'metadata':
        kernels = retrieve_kernel_metadata('tensor')
        create_csv(kernels, csv_path)
    elif args.task == 'download':
        write_property('DEFAULT', 'current_kernel', '1', config_path)
        done = False
        while not done:
            done = download_kernels(csv_path, out_dir, config_path)
            logger.info(f'Sleeping for 60 seconds at record #{current_kernel}...')
            current_kernel = read_property('DEFAULT', 'current_kernel', config_path)
            time.sleep(60)
            logger.info('Restarting...')

if __name__ == "__main__":
    main()
