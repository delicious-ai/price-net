from enum import Enum


class Aggregation(Enum):
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


class Accelerator(Enum):
    CPU = "cpu"
    GPU = "gpu"
    MPS = "mps"


class HeuristicType(Enum):
    EVERYTHING = "everything"
    GEMINI = "gemini"
    NEAREST = "nearest"
    NEAREST_BELOW = "nearest_below"
    NEAREST_PER_GROUP = "nearest_per_group"
    NEAREST_BELOW_PER_GROUP = "nearest_below_per_group"
    WITHIN_EPSILON = "within_epsilon"
    HOUGH_REGIONS = "hough_regions"


class Precision(Enum):
    BF16_MIXED = "bf16-mixed"
    HALF_MIXED = "16-mixed"
    FULL = "32"
