from typing import Dict, Union

import numpy as np


def standard_deviation(n: int, y_sq_count: float, y_count: float) -> float:
    if n == 0:
        return 0.0
    n_inv = 1 / float(n)
    return np.sqrt(np.fabs(n_inv * (y_sq_count - (n_inv * (y_count ** 2)))))


def compute_standard_deviation_reduction(sdr: Dict[str, Union[float, int]]) -> float:
    """
    calculate the standard devitation reduction
    :param sdr:     dictionary from the findBestSplit-Function
    :return:        SDR-value (float)
    """
    n_l = sdr['total'] - sdr['righttotal']
    n_r = sdr['righttotal']
    l_s = sdr['sumtotalLeft']
    l_s_sq = sdr['sumsqtotalLeft']
    r_s = sdr['sumtotalRight']
    r_s_sq = sdr['sumsqtotalRight']
    total = float(n_l + n_r)
    base = standard_deviation(n_l + n_r, l_s_sq + r_s_sq, l_s + r_s)
    sd_l = standard_deviation(n_l, l_s_sq, l_s)
    ratio_l = n_l / total
    sd_r = standard_deviation(n_r, r_s_sq, r_s)
    ratio_r = n_r / total
    return base - (ratio_l * sd_l) - (ratio_r * sd_r)
