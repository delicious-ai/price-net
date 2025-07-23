import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import json

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

    def __init__(
            self,
            cfg: ExtractionEvaluationConfig,
            result_dir: Path,
            cache_dir: Path | None = None,
            exp_name: str | None = None,
            use_cache: bool = True
    ):
        self.exp_name = exp_name
        self.result_dir = result_dir
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self.results_path = self.result_dir / f"{self.exp_name}.json"
        self.cache_dir = self._build_cache_dir(cache_dir)
        self.cache_path = self.cache_dir / f"{self.exp_name}.json"
        self.use_cache = use_cache

        factory = ExtractorFactory(cfg.extractor_config_path)
        self.extractor = factory.build()

        self.root_dir = Path(cfg.dataset_dir)
        self.price_img_dir = self.root_dir / self.price_images
        self.price_boxes = self._read_price_boxes(self.root_dir / self.price_box_fpath)

        if not os.path.exists(self.price_img_dir):
            raise RuntimeError(f"Price boxes directory {self.price_img_dir} does not exist. Try running script: build_extraction_dataset.py")

    @staticmethod
    def _build_cache_dir(cache_dir: Path | None = None) -> Path:

        if cache_dir is None:
            script_dir = Path(__file__).resolve().parent
            cache_dir = script_dir / "cache"

        cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir



    def _read_price_boxes(self, price_box_fpath: Path) -> pd.DataFrame:
        df = pd.read_csv(price_box_fpath)
        df = df[~(pd.isnull(df[self.price_contents_col]) & pd.isnull(df[self.price_type_col]))]
        return df

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


    @staticmethod
    def get_iou_bigrams(s1: str, s2: str) -> float:
        """
        Compute the Jaccard index between two strings using character bigrams.

        Args:
            s1 (str): First input string
            s2 (str): Second input string

        Returns:
            float: Jaccard similarity score
        """

        def get_bigrams(s: str) -> set:
            s = s.lower().strip()
            return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else set()

        bigrams1 = get_bigrams(s1)
        bigrams2 = get_bigrams(s2)

        intersection = bigrams1 & bigrams2
        union = bigrams1 | bigrams2

        return len(intersection) / len(union) if union else 1.0

    def eval(self):

        iou_arr = []
        iou_bigrams_arr = []
        price_is_correct = []
        type_is_correct = []

        if self.use_cache and self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                cached_outputs = json.load(f)
        else:
            cached_outputs = {}


        for i, row in tqdm(self.price_boxes.iterrows(), total=len(self.price_boxes)):
            ground_truth_price_type = self._price_type_string_to_enum(row[self.price_type_col])
            ground_truth_price, gt_price_str = self._parse_price(row[self.price_contents_col], ground_truth_price_type)
            fname = row[self.price_bbox_id_col] + self.img_format
            img_path = self.price_img_dir / fname

            if fname in cached_outputs:
                raw_output = cached_outputs[fname]
            else:
                raw_output = self.extractor(img_path)
                cached_outputs[fname] = raw_output


            pred_price_type, pred_price = self.extractor.format(raw_output)

            _, pred_price_str = self.extractor.format_as_str(raw_output)

            price_is_correct.append(pred_price_str.lower() == gt_price_str.lower())
            type_is_correct.append(pred_price_type == ground_truth_price_type)

            iou = self.get_iou_words(pred_price_str.lower(), gt_price_str.lower())
            iou_arr.append(iou)
            iou_bigram = self.get_iou_bigrams(pred_price_str.lower(), gt_price_str.lower())
            iou_bigrams_arr.append(iou_bigram)

            if self.use_cache:
                with open(self.cache_path, "w") as f:
                    json.dump(cached_outputs, f, indent=2)

        results = {
            "price_accuracy": float(np.mean(price_is_correct)),
            "price_type_accuracy": float(np.mean(type_is_correct)),
            "mean_iou": float(np.mean(iou_arr)),
            "mean_iou_bigram": float(np.mean(iou_bigrams_arr)),
        }


        print(json.dumps(results, indent=2))
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)






def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate price extraction model.")
    parser.add_argument(
        "--extractor-cfg",
        type=str,
        required=True,
        help="Path to the extractor config YAML file."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Name of experiment"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/extraction"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = ExtractionEvaluationConfig(
        extractor_config_path=Path(args.extractor_cfg),
        dataset_dir=Path(args.dataset_dir),
        cacheing=False,
    )
    evaluator = ExtractionEvaluation(
        cfg = cfg,
        exp_name = args.exp_name,
        result_dir = Path(args.results_dir)
    )
    evaluator.eval()