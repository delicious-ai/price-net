from enum import Enum


class InputReduction(Enum):
    NONE = "none"
    CLOSEST_PER_GROUP = "closest_per_group"

    """
    NONE: We predict on all product-price pairs in every scene.
    CLOSEST_PER_GROUP: We only make a prediction for one product-price pair (the closest one) for every product group.
    """


class PredictionStrategy(Enum):
    MARGINAL = "marginal"
    JOINT = "joint"

    """
    MARGINAL: We treat each prediction independently.
    JOINT: We predict for all pairs in a scene jointly.
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
