"""
Module containing class for saving camera data
"""

import os
import numpy as np
from PIL import Image
from typing import Union, Callable


class CameraDataWriter:
    """Saves camera data."""
    def __init__(self, saving_dir: Union[str, os.PathLike],
                 dir_number: int = 0,
                 dtype: str = 'int8',
                 data_extension: str = '.jpg',
                 preprocess_func: Callable = None) -> None:
        """Initializes Camera Data Writer class.

        Parameters
        ----------
        saving_dir -- directory for saving the data

        dir_number -- directory number
            default = 0

        dtype -- output datatype
            default = 'int8'

        data_extension -- file extension
            default = '.jpg'

        preprocess_func -- function for preprocessing camera data before saving
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

    def save_data(self, data: np.ndarray):
        """Saves camera data to previously specified directory.

        Parameters
        ----------
        data -- camera data to be saved

        Raises
        ------
        OSError -- if the data path is not correct or file couldn't be created
        """
        if self.preprocess_func is not None:
            data = self.preprocess_func(data)
            
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        filename = self._create_filename()
        saving_path = self._create_saving_path(filename)

        # Saving
        img = Image.fromarray(data.astype(self.dtype))
        img.save(saving_path)


if __name__ == '__main__':
    pass

