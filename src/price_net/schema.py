from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Annotated
from typing import Literal

from price_net.enums import PriceType
from pydantic import BaseModel
from pydantic import computed_field
from pydantic import Field
from pydantic import model_validator


class Price(BaseModel, ABC):
    price_type: PriceType = Field(frozen=True)
    currency: Literal["$"] = "$"

    @computed_field
    @property
    @abstractmethod
    def unit_price(self) -> float | None:
        pass

    @computed_field
    @property
    @abstractmethod
    def price_text(self) -> str:
        pass


class StandardPrice(Price):
    price_type: Literal[PriceType.STANDARD] = PriceType.STANDARD
    amount: float

    @computed_field
    @property
    def unit_price(self) -> float:
        return self.amount

    @computed_field
    @property
    def price_text(self) -> str:
        return f"{self.currency}{self.amount:.2f}"


class BulkOfferPrice(Price):
    price_type: Literal[PriceType.BULK_OFFER] = PriceType.BULK_OFFER
    quantity: int
    total_price: float

    @computed_field
    @property
    def unit_price(self) -> float:
        return self.total_price / self.quantity

    @computed_field
    @property
    def price_text(self) -> str:
        return f"{self.quantity} / {self.currency}{self.total_price:.2f}"


class BuyNForXGetMForYPrice(Price):
    price_type: Literal[PriceType.BUY_N_FOR_X_GET_M_FOR_Y] = (
        PriceType.BUY_N_FOR_X_GET_M_FOR_Y
    )
    buy_quantity: int
    buy_price: float | None = None
    get_quantity: int
    get_price: float | None = None

    @computed_field
    @property
    def unit_price(self) -> float | None:
        total_items = self.buy_quantity + self.get_quantity
        if self.buy_price is not None and self.get_price is not None:
            total_cost = self.buy_price + self.get_price
            return total_cost / total_items if total_items else None
        else:
            return None

    @computed_field
    @property
    def price_text(self) -> str:
        return (
            f"Buy {self.buy_quantity} for {self.currency}{self.buy_price:.2f}, "
            f"Get {self.get_quantity} for {self.currency}{self.get_price:.2f}"
        )


class UnknownPrice(Price):
    price_type: Literal[PriceType.UNKNOWN] = PriceType.UNKNOWN

    @computed_field
    @property
    def unit_price(self) -> None:
        return None

    @computed_field
    @property
    def price_text(self) -> Literal["Unknown"]:
        return "Unknown"


class MiscPrice(Price):
    price_type: Literal[PriceType.MISC] = PriceType.MISC
    contents: str

    @computed_field
    @property
    def unit_price(self) -> None:
        return None

    @computed_field
    @property
    def price_text(self) -> str:
        return self.contents


PriceModelType = Annotated[
    StandardPrice | BulkOfferPrice | BuyNForXGetMForYPrice | UnknownPrice | MiscPrice,
    Field(discriminator="price_type"),
]


class ProductPrice(BaseModel):
    upc: str
    price: PriceModelType


class BoundingBox(BaseModel):
    cx: float = Field(ge=0.0, le=1.0)
    cy: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)


class PriceGroup(BaseModel):
    group_id: str
    product_bbox_ids: set[str]
    price_bbox_ids: set[str]


class ProductGroup(BaseModel):
    group_id: str
    product_bbox_ids: set[str]


class PriceAttributionScene(BaseModel):
    scene_id: str
    product_bboxes: dict[str, BoundingBox]
    price_bboxes: dict[str, BoundingBox]
    prices: dict[str, PriceModelType]
    products: dict[str, str]
    product_groups: list[ProductGroup]
    price_groups: list[PriceGroup]
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def check_ids_are_consistent(self) -> PriceAttributionScene:
        product_bbox_ids = set(self.product_bboxes.keys())
        price_bbox_ids = set(self.price_bboxes.keys())
        if product_bbox_ids & price_bbox_ids:
            raise ValueError("Product and price bbox IDs should be disjoint.")
        prod_ids_already_in_group = set()
        for product_group in self.product_groups:
            if product_group.product_bbox_ids & prod_ids_already_in_group:
                raise ValueError(
                    "Product groups should be disjoint (1 group assignment / prod. bbox)."
                )
            prod_ids_already_in_group.update(product_group.product_bbox_ids)
            if product_group.product_bbox_ids - product_bbox_ids:
                raise ValueError(
                    f"Prod. group {product_group.group_id} has IDs not found in the scene product bbox IDs."
                )
        prod_id_diff = set(self.product_bboxes.keys()).difference(
            set(self.products.keys())
        )
        if prod_id_diff:
            raise ValueError(
                "The IDs specified for `products` should match the ones for `product_bboxes`."
            )

        price_ids_already_in_group = set()
        for price_group in self.price_groups:
            if price_group.price_bbox_ids & price_ids_already_in_group:
                raise ValueError(
                    "Price groups should be disjoint (1 group assignment / price bbox)."
                )
            price_ids_already_in_group.update(price_group.price_bbox_ids)
            if price_group.product_bbox_ids - product_bbox_ids:
                raise ValueError(
                    f"Price group {price_group.group_id} has product IDs not found in the scene-level product bbox IDs."
                )
            if price_group.price_bbox_ids - price_bbox_ids:
                raise ValueError(
                    f"Price group {price_group.group_id} has price IDs not found in the scene-level price bbox IDs."
                )

        price_id_diff = set(self.price_bboxes.keys()).difference(
            set(self.prices.keys())
        )
        if price_id_diff:
            raise ValueError(
                "The IDs specified for `prices` should match the ones for `price_bboxes`."
            )
        return self
