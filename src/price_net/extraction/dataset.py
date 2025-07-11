from pathlib import Path
from collections import defaultdict
import pandas as pd

from torch.utils.data import Dataset

from price_net.enums import PriceType
from price_net.association.transforms import ConcatenateBoundingBoxes, InputTransform


class PriceExtractDataset(Dataset):

    price_box_fpath = "price_boxes.csv"
    price_type_col = "price_type"
    price_contents_col = "price_contents"

    def __init__(self, root_dir: str | Path):
        """Initialize a `PriceAssociationDataset`.

        Args:
            root_dir (str | Path): Root directory where the dataset is stored.
            input_transform (InputTransform, optional): Transform to apply to each input of the dataset when `__getitem__` is called. Defaults to `ConcatenateBoundingBoxes`.
            aggregation (Aggregation, optional): Determines how we parse instances for `__getitem__`. Defaults to Aggregation.NONE (each potential product-price association pair is returned for a scene).
            use_depth (bool, optional): Whether/not to use depth if aggregating by "closest_per_group".
        """
        self.root_dir = Path(root_dir)
        self.price_img_dir = self.root_dir / "images"
        self.price_boxes = pd.read_csv(self.root_dir / self.price_box_fpath)


        self.instances = self._get_instances(self.price_boxes)

    def _price_type_string_to_enum(self, price_type:str) -> PriceType:
        """
        Converts a price type string into its corresponding PriceType enum value.

        This method maps specific price type strings (such as "BUY_X_GET_Y" and
        "SINGLE_SALE") to their respective PriceType enum values. If the provided
        price type string does not match any predefined mappings, it will be
        directly converted to a PriceType enum.

        Args:
            price_type (str): The price type as a string.

        Returns:
            PriceType: The corresponding enum value for the provided price type string.
        """
        if price_type == "BUY_X_GET_Y":
            return PriceType.BUY_X_GET_Y_FOR_Z
        elif price_type == "SINGLE_SALE":
            return PriceType.STANDARD
        else:
            return PriceType(price_type)

    def _get_price(self, price_contents: str, price_type: PriceType) -> pd.DataFrame:

        if price_type == PriceType.BUY_X_GET_Y_FOR_Z:
            pass
        elif price_type == PriceType.STANDARD:
            pass
        elif price_type == PriceType.UNKNOWN:
            pass
        else:
            raise ValueError(f"Unknown price type: {price_type}")

    def _get_instances(self, price_boxes: pd.DataFrame) -> dict:
        instances = {}
        for i, (_, vals) in enumerate(price_boxes.iterrows()):

            price_type = self._price_type_string_to_enum(vals[self.price_type_col])
            price = self._get_price(vals[self.price_contents_col], price_type)

            instances[i] = {
                "price_type": price_type,
                "price": price

            }



if __name__ == "__main__":

    data_dir = Path("/Users/porterjenkins/data/price-attribution-scenes/test")

    dataset = PriceExtractDataset(root_dir=data_dir)