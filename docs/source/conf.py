"""Sphinx configuration for the tda-finance-mapper documentation."""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.abspath("../../src"))


project = "tda-finance-mapper"
author = "Jaime Corzo Galdó"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

html_css_files = [
    "custom.css",
]

