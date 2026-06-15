Installation
============

There are two ways to install the package.

Install from PyPI
-----------------

This is the recommended option if you only want to use the package.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install tda-finance-mapper

On Windows:

.. code-block:: bash

   python -m venv .venv
   .venv\Scripts\activate
   pip install tda-finance-mapper

Install from the repository
---------------------------

This option is recommended if you want to reproduce the experiments, inspect the
source code, modify the package or build the documentation.

.. code-block:: bash

   git clone https://github.com/jaimecrz3/tda-finance-mapper.git
   cd tda-finance-mapper
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

On Windows:

.. code-block:: bash

   git clone https://github.com/jaimecrz3/tda-finance-mapper.git
   cd tda-finance-mapper
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

The ``requirements.txt`` file installs the project in editable mode with:

.. code-block:: text

   -e .

This means that changes made in the source code are immediately available
without reinstalling the package.

Development dependencies
------------------------

For development tools such as ``flake8``, ``pytest``, ``sphinx``, ``build`` and
``twine``, install the optional development dependencies:

.. code-block:: bash

   pip install -e .[dev]
