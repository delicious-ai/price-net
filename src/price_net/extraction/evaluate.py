import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import string

from price_net.enums import PriceType

from price_net.extraction.configs import ExtractionEvaluationConfig
from price_net.extraction.factory import ExtractorFactory
from price_net.extraction.parsers import *

class ExtractionEvaluation(object):

    price_box_fpath = "price_boxes.csv"
    price_type_col = "price_type"
    price_contents_col = "price_contents"
    price_images = "price-images"
    price_bbox_id_col = "price_bbox_id"
    img_format = ".jpg"

    def __init__(self, cfg: ExtractionEvaluationConfig):
        factory = ExtractorFactory(cfg.extractor_config_path)
        self.extractor = factory.build()

        self.root_dir = Path(cfg.dataset_dir)
        self.price_img_dir = self.root_dir / self.price_images
        self.price_boxes = pd.read_csv(self.root_dir / self.price_box_fpath)

        if not os.path.exists(self.price_img_dir):
            raise RuntimeError(f"Price boxes directory {self.price_img_dir} does not exist. Try running script: build_extraction_dataset.py")


    def _price_type_string_to_enum(self, price_type: str) -> PriceType:
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

    def _parse_price(self, price_contents: str, price_type: PriceType) -> pd.DataFrame:

        if price_type == PriceType.STANDARD:
            return parse_regular_price(price_contents)
        elif price_type == PriceType.BULK_OFFER:
            return parse_bulk_offer_price(price_contents)
        elif price_type == PriceType.BUY_X_GET_Y_FOR_Z:
            return parse_buy_x_get_y_price(price_contents)
        elif price_type == PriceType.UNKNOWN:
            return parse_unreadable_price(price_contents)
        else:
            raise ValueError(f"Unknown price type: {price_type}")

    @staticmethod
    def get_iou_words(str1: str, str2: str) -> float:
        """
            Compute the Jaccard index between two strings based on word tokens,
            after removing '/' and ','.
        """
        for ch in ["/", ","]:
            str1 = str1.replace(ch, "")
            str2 = str2.replace(ch, "")


        set1 = set(str1.split())
        set2 = set(str2.split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        iou = len(intersection) / len(union) if union else 1.0
        return iou

    def eval(self):

        iou_arr = []
        price_is_correct = []
        type_is_correct = []



        for i, row in tqdm(self.price_boxes.iterrows(), total=len(self.price_boxes)):
            ground_truth_price_type = self._price_type_string_to_enum(row[self.price_type_col])
            ground_truth_price, gt_price_str = self._parse_price(row[self.price_contents_col], ground_truth_price_type)
            fname = row[self.price_bbox_id_col] + self.img_format
            img_path = self.price_img_dir / fname
            raw_output = self.extractor(img_path)
            pred_price_type, pred_price = self.extractor.format(raw_output)

            _, pred_price_str = self.extractor.format_as_str(raw_output)

            price_is_correct.append(pred_price_str == gt_price_str)
            type_is_correct.append(pred_price_type == ground_truth_price_type)

            iou = self.get_iou_words(pred_price_str, gt_price_str)
            iou_arr.append(iou)



        print("Price Accuracy: ", np.mean(price_is_correct))
        print("Price Type Accuracy: ", np.mean(type_is_correct))
        print("mIoU: ", np.mean(iou_arr))



if __name__ == "__main__":
    cfg = ExtractionEvaluationConfig()
    evaluator = ExtractionEvaluation(cfg)
    evaluator.eval()