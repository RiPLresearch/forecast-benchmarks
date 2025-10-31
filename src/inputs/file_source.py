import os
from typing import Any
import pandas as pd
from src.constants import PATHS
from src.data import Number
from src.utils import read_json, read_pickle


class FileDataSource:
    """
    Returns data from local cache. If data does not exist,
    returns and saves data queried from the fallback datasource.

    If the fallback datasource is not defined, requests for uncached data will
    raise an error.
    """
    cache_path = PATHS.cache_path()

    def _get_file_utility(self,
                          data_type: str,
                          patient_id: str,
                          file_type: str = 'json') -> Any:
        """
        Utility function try loading data from cache, otherwise queries
        fallback data source, writes data source to json and returns data

        Parameters
        ----------
        data_type: str
            data type name used to read file data and
            call fallback data source function to get data type
        patient_id: str
        file_type: str
            File type of the data. Current supports 'json' for seizure events.
        """
        data_file_path = os.path.join(self.cache_path, data_type,
                                      f'{patient_id}.{file_type}')
        try:
            # Read data from local cache
            if file_type == 'json':
                return read_json(data_file_path)
            elif file_type == 'pickle':
                return read_pickle(data_file_path)
        except:
            print(f'Unable to open {data_file_path} from local cache.')
            return None

    def get_seizure_events(self,
                         patient_id: str,
                         _from_time: Number = 0,
                         _to_time: Number = 9e12) -> Any:
        return self._get_file_utility('seizure_events', patient_id)

