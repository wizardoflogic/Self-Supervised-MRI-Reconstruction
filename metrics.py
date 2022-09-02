import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from skfuzzy import nmse


class Metrics:
    """
    Class to compute metrics and visualize data
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_nmse(
        img1: np.ndarray,
        img2: np.ndarray
    ) -> float:
        """
        Computes normalized mean squared error.

        see normalized_root_mse docs: https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.normalized_root_mse

        Parameters
        ----------
        img1: Image 1.
        img2: Image 2.

        Returns
        -------
        Normalized mean squared error.
        """
        return nmse(img1,img2)

    @staticmethod
    def compute_nrmse(
        img1: np.ndarray,
        img2: np.ndarray
    ) -> float:
        """
        Computes normalized root mean squared error.

        See normalized_root_mse docs: https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.normalized_root_mse.

        Parameters
        ----------
        img1: Image 1.
        img2: Image 2.

        Returns
        -------
        Normalized root mean squared error.
        """
        return nrmse(img1,img2,normalization='Euclidean')

    @staticmethod
    def compute_ssim(
        img1: np.ndarray,
        img2: np.ndarray
    ) -> float:
        """
        Computes structural similarity index.

        Parameters
        ----------
        img1: Image 1.
        img2: Image 2.

        Returns
        -------
        Structural similarity index.
        """
        return ssim(img1,img2,multichannel=True)
