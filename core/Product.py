import itertools

class Product:
    def __init__(self, name: str, volume: float, price: float) -> None:
        self._name: str = name
        self._volume: float = volume
        self._price: float = price

    @property
    def name(self) -> str:
        return self._name

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def price(self) -> float:
        return self._price

    def __repr__(self) -> str:
        return f"Product(name='{self._name}', volume={self._volume}, price={self._price})"
