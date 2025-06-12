import csv

def record_error(ref: str, error: str, path: str = 'logs.csv'):
    """
    Records an error message to a CSV log file.

    Args:
        ref (str): A reference identifier for the error.
        error (str): The error message to be logged.
        path (str, optional): The file path to the CSV log file. Defaults to 'logs.csv'.

    Returns:
        None
    """
    with open(path, mode='a', newline='') as report_file:
        fieldnames = ['ref', 'error']
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writerow({'ref': ref, 'error': error})