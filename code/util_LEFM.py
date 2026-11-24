import numpy as np


def K1(T, l):
    term1 = (1 - T) * np.sqrt(l)
    term2 = (2 / np.pi) * T * np.sqrt((l - 1) / l)
    term3 = (2 * T / np.pi) * np.sqrt(l) * np.arctan(np.sqrt(1 / (l - 1)))
    return term1 - term2 + term3

def power_Kc_L(T, l):
    sqrt_lm1 = np.sqrt(l - 1)
    inv_sqrt_lm1 = np.sqrt(1 / (l - 1))
    arccot_lm1 = np.arctan(1 / sqrt_lm1)

    numerator = (
        2 * T
        + sqrt_lm1 * l * (np.pi * (-1 + T) + 2 * inv_sqrt_lm1 * T)
        - (2 * l * T * arccot_lm1) / inv_sqrt_lm1
    )

    denominator = (
        2 * sqrt_lm1 * l * np.pi * (-1 + T)
        - 4 * T
        + 4 * l * T
        - (4 * l * T * arccot_lm1) / inv_sqrt_lm1
    )

    return numerator / denominator
