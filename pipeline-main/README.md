# On the Reproducibility of Jupyter Notebooks

The current repository contains the script of the initial pipeline used to collect Jupyter notebooks relying on tensors from Kaggle using the Kaggle API. This is to be used for research and academic purposes.

## Usage
This script helps manage Kaggle kernels by retrieving metadata or downloading kernel files.

### General Command Structure
```sh
python script.py <task>
```
- `<task>`: Choose between **`metadata`** (to fetch kernel metadata) or **`download`** (to download kernels).

### Running the Metadata Task
To retrieve kernel metadata and save it as a CSV file:
```sh
python script.py metadata
```
You will be prompted to enter:
- **CSV file path** *(Default: `jupyter-nbs/data/nbs.csv`)*

### Running the Download Task
To download kernels using the CSV file:
```sh
python script.py download
```
You will be prompted to enter:
- **CSV file path** *(Default: `jupyter-nbs/data/nbs.csv`)*
- **Output directory** *(Default: `jupyter-nbs/data/nbs/`)*
- **Configuration file path** *(Default: `jupyter-nbs/data/config.properties`)*

### How the Script Works
- Reads the CSV file for kernel details.
- Uses the config file to track progress.
- Downloads kernels in intervals of **60 seconds**.
- Logs progress and restarts automatically if needed.

## Contributing
We welcome contributions! Please fork the repo and submit a pull request. You can also keep your own variant via a fork nd work independently from this project.

## License
This project is licensed under the MIT License.

## Contact
For questions, reach out to [l.m.ochoa.venegas@tue.nl](mailto:l.m.ochoa.venegas@tue.nl).
