# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to the Python path for autodoc
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
project = "Neural Network"
copyright = "2025, Danilo Carneiro de Souza"
author = "Danilo Carneiro de Souza"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/NumPy docstring support
    "sphinx.ext.viewcode",  # Source code links
    "sphinx.ext.intersphinx",  # Links to other documentation
    "sphinx.ext.mathjax",  # Math rendering
    "sphinx.ext.githubpages",  # GitHub Pages support
    "sphinx.ext.doctest",  # Test code examples
    "sphinx.ext.coverage",  # Documentation coverage
]

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"

# Intersphinx mapping for external documentation links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Template and exclusion settings
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file settings
source_suffix = {
    ".rst": None,
}
master_doc = "index"

# Language settings
language = "en"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Custom CSS
html_css_files = []

# HTML context
html_context = {
    "display_github": True,
    "github_user": "daniloldn",
    "github_repo": "Neural-network",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# Custom sidebar
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {}

# Grouping the document tree into LaTeX files
latex_documents = [
    (
        master_doc,
        "neural-network.tex",
        "Neural Network Documentation",
        "Danilo Carneiro de Souza",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, "neural-network", "Neural Network Documentation", [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        master_doc,
        "neural-network",
        "Neural Network Documentation",
        author,
        "neural-network",
        "A custom neural network implementation.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_exclude_files = ["search.html"]
