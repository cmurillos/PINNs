"""Hace importables los modulos del repo (raiz) durante los tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
