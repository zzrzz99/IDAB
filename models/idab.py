"""
IDAB module — stable import path ``models.idab``.

Implementation is maintained in ``idab_v2.py``; this file re-exports ``IDAB`` so
``from models.idab import IDAB`` works after rename workflows without duplicating code.
"""
from .idab_v2 import IDAB

__all__ = ["IDAB"]
