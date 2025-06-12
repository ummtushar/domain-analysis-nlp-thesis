import csv
from kaggle.models.kaggle_models_extended import Kernel
from utils.logger import record_error
import sys
from utils.kaggle_api import get_kaggle_api_instance

api = get_kaggle_api_instance()

def _fetch_nb_metadata(page: int, search_str: str) -> list[Kernel]:
    """
    Fetches metadata for Jupyter notebooks from the API.

    Args:
        page (int): The page number to fetch.
        search_str (str): The search string to filter the notebooks.

    Returns:
        list[Kernel]: A list of Kernel objects containing the metadata of the notebooks.
    """
    return api.kernels_list(page=page,
                            page_size=50,
                            search=search_str)

def retrieve_kernel_metadata(search_str: str) -> list[Kernel]:
    """
    Retrieve metadata for kernels matching the search string.

    This function fetches metadata for kernels from a paginated API, 
    iterating through pages until no more kernels are found.

    Args:
        search_str (str): The search string to filter kernels.

    Returns:
        list[Kernel]: A list of Kernel objects containing metadata.
    """
    page = 1
    nbs = []
    while True:
        kernels = _fetch_nb_metadata(page, search_str)
        if kernels == []:
            break
        nbs.extend(kernels)
        page += 1
    return nbs

def create_csv(kernels: list[Kernel], path: str = None) -> None:
    """
    Creates a CSV file from a list of Kernel objects.
    Args:
        kernels (list[Kernel]): A list of Kernel objects to be written to the CSV file.
        path (str, optional): The file path where the CSV file will be saved. If None, the CSV content will be written to stdout. Defaults to None.
    Returns:
        None
    """
    ref = None
    fields = ['id', 'ref', 'user', 'slug', 'title', 'author', 'lastRunTime', 'totalVotes']

    if path is None:
        writer = csv.writer(sys.stdout)
    else:
        file = open(path, 'w')
        writer = csv.writer(file)
            
    writer.writerow(fields)
    _id = 1
    for kernel in kernels:
        try:
            ref = str(getattr(kernel, 'ref'))
            user, slug = ref.split('/')
            kernel_fields = [_id, ref, user, slug] + [str(getattr(kernel, field)) for field in fields if field not in ['ref', 'user', 'slug']]
            writer.writerow(kernel_fields)
            _id += 1
        except Exception as e:
            record_error(ref, e)
            continue
            
    if path is not None:
        file.flush()
        file.close()
