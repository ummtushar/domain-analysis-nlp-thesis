import configparser

config = configparser.ConfigParser()

def read_property(section: str, property: str, config_path: str) -> str:
    """
    Reads a property value from a configuration file.

    Args:
        section (str): The section in the configuration file.
        property (str): The property name within the section.
        config_path (str): The path to the configuration file.

    Returns:
        str: The value of the specified property.
    """
    config.read(config_path)
    return config[section][property]

def write_property(section: str, property: str, value: any, config_path: str) -> None:
    """
    Writes a property to a specified section in a configuration file.

    If the section does not exist, it will be created.

    Args:
        section (str): The section in the configuration file where the property will be written.
        property (str): The property name to be written.
        value (any): The value of the property to be written.
        config_path (str): The path to the configuration file.

    Returns:
        None
    """
    config.read(config_path)
    if section not in config:
        config.add_section(section)
    config.set(section, property, str(value))

    with open(config_path, 'w') as config_file:
        config.write(config_file)
