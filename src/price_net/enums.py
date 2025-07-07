from enum import Enum


class InputGranularity(Enum):
    PAIRWISE = "pairwise"
    SCENE_LEVEL = "scene_level"

    """
    - PAIRWISE: Each input is a single candidate product-price pair.
    - SCENE_LEVEL: Each input is a full scene (set) of product-price candidate pairs.
    """


class FeaturizationMethod(Enum):
    CENTROID = "centroid"
    CENTROID_DIFF = "centroid_diff"

    """
    - CENTROID: The first 3 features of our "association vector" are the centroid of the product in the group nearest to the specified price tag.
    - CENTROID_DIFF: The first 3 features of our "association vector" are the difference between the product and price tag centroids.
    """


class PriceType(str, Enum):
    # Single item price
    STANDARD = "STANDARD"
    # Buy multiple to get sale (3 for $5.00)
    BULK_OFFER = "BULK_OFFER"
    # Covers BOGO, "buy 3 for $3, get 1 for $1.50", etc.
    BUY_X_GET_Y_FOR_Z = "BUY_X_GET_Y_FOR_Z"
    # Catch all for other types of pricing
    MISC = "MISC"
    # Price tag is too blurry or otherwise unreadable
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_value(cls, value):
        _int_map = {
            0: cls.UNKNOWN,
            1: cls.STANDARD,
            2: cls.BULK_OFFER,
            3: cls.BUY_X_GET_Y_FOR_Z,
            4: cls.MISC,
        }
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return _int_map.get(value, cls.UNKNOWN)
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                try:
                    return cls[value]
                except KeyError:
                    return cls.UNKNOWN
        return cls.UNKNOWN
