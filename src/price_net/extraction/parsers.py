import torch


def parse_regular_price(price_contents: str) -> torch.Tensor:
    """
    Parses a regular price string and converts it into a tensor format.
        - Example: "$4.99" -> tensor([4.9900])
        - ndims: 1

    Args:
        price_contents (str): A string representing the price, prefixed with a
            dollar sign (e.g., "$19.99").

    Returns:
        torch.Tensor: (float) A tensor containing the numeric representation of the price
            extracted from the input string.
    """
    price = float(price_contents.strip("$"))
    output = torch.tensor([price])
    return output


def parse_volume_discount_price(price_contents: str) -> torch.Tensor:
    """
    Parses a formatted string containing volume and price information and converts it
    into a tensor with numeric values for further processing or computation.
        - Example: "5 / $40" -> tensor([ 5., 40.])
        - ndims: 2

    Args:
        price_contents: A string containing the volume (units) and the corresponding
            price separated by a "/" symbol. Example format: "10 / $25".

    Returns:
        torch.Tensor: (float) A tensor containing two elements. The first element is the
            parsed floating-point value of the units, and the second element is the
            parsed floating-point value of the price.
    """
    units, price = price_contents.split("/")
    units = float(units.strip())
    price = float(price.strip().strip("$"))
    output = torch.tensor([units, price])
    return output


def parse_unreadable_price(price_contents: str) -> torch.Tensor:
    return torch.tensor([0.0])


def parse_buy_x_get_y_price(price_contents: str) -> torch.Tensor:
    """
    Parses a string representing a "Buy X Get Y for Z" price structure and converts
    the numeric values into a PyTorch tensor.
        - Example: Buy 1, Get 1 / $0 - > tensor([1., 1., 0.])
        - ndims: 3

    This function takes an input string in the format "Buy X get Y for Z", where X
    is the quantity of items to be purchased, Y is the quantity of additional items
    to be acquired at the specified discounted price Z. It extracts the numeric
    values for X, Y, and Z, converts them to floating-point numbers, and packs them
    into a PyTorch tensor.

    Args:
        price_contents (str): A string describing a "Buy X get Y for Z" price
            structure. The string should be formatted as "Buy X get Y for $Z".

    Returns:
        torch.Tensor: (float) A tensor containing the numeric values of X, Y, and Z in
        the "Buy X get Y for Z" structure.
    """
    x, y_price = price_contents.lower().split("get")
    x = float(x.split(" ")[1].strip().strip(","))
    y, price = y_price.split("/")

    y = float(y.strip())
    price = float(price.strip().strip("$"))

    return torch.tensor([x, y, price])

def parse_generic_price(price_contents: str) -> torch.Tensor:
    pass



if __name__ == "__main__":

    print(parse_regular_price(price_contents="$4.99"))
    print(parse_volume_discount_price(price_contents="5 / $40"))
    print(parse_unreadable_price("Unreadable"))
    print(parse_buy_x_get_y_price(price_contents="Buy 1, Get 1 / $0"))