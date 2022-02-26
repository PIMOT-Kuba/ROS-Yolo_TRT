"""
Module containing class for saving lidar data
"""

import os

from typing import Union, Callable
import numpy as np
import sensor_msgs.point_cloud2 as pc2


class LidarDataWriter:
    """Saves lidar data in a given format."""
    def __init__(self, saving_dir: Union[str, os.PathLike],
                 dtype: str = 'float32',
                 preprocess_func: Callable = None) -> None:
        """Initializes Lidar Data Writer class.

        Parameters
        ----------
        saving_dir -- directory for saving the data

        dtype -- output datatype
            default = 'float32'

        preprocess_func -- function for preprocessing lidar data before saving
            default = None
        """
        try:
            np.dtype(dtype)
            self.dtype = dtype
        except TypeError:
            raise TypeError('Incorrect dtype for numpy array')
        self.saving_dir = saving_dir
        self.preprocess_func = preprocess_func

    def preprocess(lidar_data):
        x = np.array(list(pc2.read_points(lidar_data, skip_nans=True, field_names="x")))
        y = np.array(list(pc2.read_points(lidar_data, skip_nans=True, field_names="y")))
        z = np.array(list(pc2.read_points(lidar_data, skip_nans=True, field_names="z")))
        p = np.array(list(pc2.read_points(lidar_data, skip_nans=True, field_names="intensity")))[:, 0]

        points_array = np.zeros(x.shape[0] + y.shape[0] + z.shape[0] + p.shape[0], dtype=np.float32)
        points_array[::4] = np.squeeze(x)
        points_array[1::4] = np.squeeze(y)
        points_array[2::4] = np.squeeze(z)
        points_array[3::4] = p
        return points_array

    def save_data(self, data: np.ndarray, saving_path: str) -> None:
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

        if self.preprocess_func is not None:
            data = self.preprocess_func(data)

        # Saving
        data.astype(self.dtype).tofile(saving_path)
