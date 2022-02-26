"""
Module containing class for saving lidar data
"""

import os

from typing import Union, Callable
import numpy as np


class LidarDataWriter:
    """Saves lidar data in a given format."""
    def __init__(self, saving_dir: Union[str, os.PathLike],
                 dir_number: int = 0,
                 dtype: str = 'float32',
                 data_extension: str = '.bin',
                 preprocess_func: Callable = None) -> None:
        """Initializes Lidar Data Writer class.

        Parameters
        ----------
        saving_dir -- directory for saving the data

        dir_number -- directory number
            default = 0

        dtype -- output datatype
            default = 'float32'

        data_extension -- file extension
            default = '.bin'

        preprocess_func -- function for preprocessing lidar data before saving
            default = None
        """
        self.saving_dir = saving_dir
        self.samples_saved = 0
        self.dir_number = dir_number
        try:
            np.dtype(dtype)
            self.dtype = dtype
        except TypeError:
            raise TypeError('Incorrect dtype for numpy array')
        self.data_extension = data_extension
        self.preprocess_func = preprocess_func

    def _create_filename(self) -> str:
        """Creates a filename for a data sample by concatena"""
        return '_'.join((str(self.dir_number), str(self.samples_saved), str(self.data_extension)))

    def _create_saving_path(self, filename: Union[str, os.PathLike]) -> str:
        """Concatenates the saving dir path and the filename for a data sample."""
        return os.path.join(self.saving_dir, filename)

    def save_data(self, data: np.ndarray) -> None:
        """Saves lidar data to previously specified directory.

        Parameters
        ----------
        lidar_data -- lidar data to be saved

        Raises
        ------
        OSError -- if the data path is not correct or file couldn't be created
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        filename = self._create_filename()
        saving_path = self._create_saving_path(filename)

        if self.preprocess_func is not None:
            data = self.preprocess_func(data)

        # Saving
        data.astype(self.dtype).tofile(saving_path)
