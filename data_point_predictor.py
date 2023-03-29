#!/bin/python3

from typing import List, Optional, Tuple
from dataclasses import dataclass
from math import sqrt
from copy import deepcopy

#
# Complete the 'calcMissing' function below.
#
# The function accepts STRING_ARRAY readings as parameter.
#

LOOK_AHEAD: int = 100
INCLUSION_THRESHOLD: float = 0.8


def extract_data(s: str) -> Tuple[str, Optional[float]]:
    """Extracts the date and value from a string.
    Returns None if the value is None."""
    dt, val = s.split("\t")

    if "None" in val:
        return dt, None
    return dt, float(val)


@dataclass(init=True, frozen=True)
class Line:
    gradient: float
    y_intercept: float


def lines_in_set(x: List[float], y: List[float]) -> List[Line]:
    """Return the best fit line for each prefix of x and y. This uses least
    squares regression."""
    x_sum: float = 0
    y_sum: float = 0
    x_sq: float = 0
    xy: float = 0
    lines: List[Line] = []

    for i, x_val in enumerate(x):
        x_sum += x_val
        y_sum += y[i]
        x_sq += x_val**2
        xy += x_val * y[i]

        offset: int = i + 1

        if ((offset * x_sq) - (x_sum**2)) != 0:
            gradient: float = ((offset * xy) - (x_sum * y_sum)) / (
                (offset * x_sq) - (x_sum**2)
            )
            y_int: float = (y_sum / offset) - gradient * (x_sum / offset)
            lines.append(Line(gradient, y_int))

    return lines


def best_fit_index(snd_gradients: List[float]) -> int:
    """Return the index of the element in snd_gradients that best fits the dataset.
    This algorithm gives preference to the first elements in the list (closest to the
    target point)."""
    s: float = 0
    s_sq: float = 0
    outliers: int = 0
    cont_outliers: int = 0

    for i, g in enumerate(snd_gradients, start=1):
        count: int = i - outliers
        s += g
        s_sq += g**2
        mean: float = s / count
        std_dev: float = sqrt((s_sq / count) - (mean**2))

        # If the element is too far from the mean of the first i elements,
        # it is considered an outlier and not included. 2 outliers in a row
        # will stop the regression.
        if abs(g - mean) > INCLUSION_THRESHOLD * std_dev:
            cont_outliers += 1
            outliers += 1
            s -= g
            s_sq -= g**2
            if cont_outliers >= 2:
                return i - 3
            continue

        cont_outliers = 0

    return len(snd_gradients) - 1


def regress_from(readings: List[str], i: int, forward: bool = True) -> Optional[Line]:
    """Returns the line that best matches the points ahead of or behind of the index i.

    forward indicates if this algorithm should check the points before or ahead of the ith
    element."""
    dr: int = 2 * forward - 1  # Direction
    count: int = 1 * dr
    j: int = i
    missing_cnt: int = 0
    nums: List[float] = []

    limit: int = LOOK_AHEAD + 1

    # Collect the values to calculate the gradient of.
    while (
        (-limit < count < limit)
        and (0 <= (j + count) < len(readings))
        and (missing_cnt < 2)
    ):
        _, val = extract_data(readings[j + count])

        # If there are 2 missing values in a row, give up
        if val is None:
            missing_cnt += 1
            j += 1
            continue
        missing_cnt = 0

        nums.append(float(val))
        count += 1 * dr

    # Cannot get the gradient of nothing or 1 point
    if len(nums) <= 1:
        return None

    lines: List[Line] = lines_in_set(list(range(dr, count, dr)), nums)
    snd_derivative: List[float] = list(
        map(
            lambda x: x.gradient,
            lines_in_set(
                list(range(dr, dr * len(lines) + dr, dr)),
                list(map(lambda y: y.gradient, lines)),
            ),
        )
    )

    return lines[best_fit_index(snd_derivative)]


def calcMissing(readings: List[str]) -> List[str]:
    """I have created a segmented linear regression algorithm to find missing data points.
    It uses 2 linear regressions to estimate the location of the missing data point.

    This works by calculating the gradient of points before and after the missing one,
    and following the found gradients. It uses the 2nd derivative to try and find a segment
    for which a linear pattern in the data holds (however this requires some tweaking).
    (Eg: A value close to 0 in the 2nd derivative indicates that there is little change in the gradient,
     thus a constant pattern is holding.)
    """
    # It isn't good practice to modify input.
    readings_cpy: List[str] = deepcopy(readings)
    res: List[Optional[float]] = [None] * 20
    locations: List[int] = [0] * 20

    # Find the location of every None value
    j: int = 0
    for i, reading in enumerate(readings_cpy):
        _, val = extract_data(reading)
        if val is None:
            locations[j] = i
            j += 1

    # Fill in the missing values when possible.
    # Sometimes it isn't possible to fill a missing value because there
    # is too little info, so it is skipped and returned to later when there is more info.
    while None in res:
        for i, l in enumerate(locations):
            dt, val = extract_data(readings_cpy[l])

            if val is None:
                # Do linear regression to and from the current point
                line_r: Optional[Line] = regress_from(readings_cpy, l, forward=True)
                line_l: Optional[Line] = regress_from(readings_cpy, l, forward=False)

                if line_r is None:
                    line_r = line_l
                if line_l is None:
                    line_l = line_r
                if line_l is line_r is None:
                    continue

                # The average of the two regressions is the estimation.
                # TODO: This is not a great way of deriviting an estimation
                estimation: float = round((line_r.y_intercept + line_l.y_intercept) / 2, 2)
                readings_cpy[l] = f"{dt}\t{estimation}"
                res[i] = estimation

    return readings_cpy
