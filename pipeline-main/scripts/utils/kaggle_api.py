from kaggle.api.kaggle_api_extended import KaggleApi

api_instance = None

def get_kaggle_api_instance() -> KaggleApi:
    """
    Returns an authenticated instance of the KaggleApi.
    This function ensures that only one instance of the KaggleApi is created and authenticated.
    If the instance does not exist, it initializes and authenticates it. If it already exists,
    it simply returns the existing instance.
    Returns:
        KaggleApi: An authenticated instance of the KaggleApi.
    """
    global api_instance

    if api_instance is None:
        api_instance = KaggleApi()
        api_instance.authenticate()
        
    return api_instance
