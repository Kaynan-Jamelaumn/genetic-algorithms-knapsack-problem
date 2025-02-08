import itertools

class Product:
    def __init__(self, name, volume, price):
        self._name = name
        self._volume = volume
        self._price = price

    @property
    def name(self):
        return self._name

    @property
    def volume(self):
        return self._volume

    @property
    def price(self):
        return self._price

    def __repr__(self):
        return f"Product(name='{self._name}', volume={self._volume}, price={self._price})"
