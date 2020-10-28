.. _installation:

=========================
Download and installation
=========================

Requirements
============

Gisele has been tested on Windows.

Running Gisele requires four things:

1. The Python programming language, version 3.7 or higher.
2. A number of Python add-on modules .
3. A solver: Gisele has been tested with CBC, GLPK, Gurobi, and CPLEX. Any other solver that is compatible with Pyomo should also work.
4. The Gisele software itself.

Recommended installation method
===============================

Clone or download the GitHub project folder.
The easiest way to get a working Gisele installation is to use the free ``conda`` package manager.

To get ``conda``, `download and install the "Miniconda" distribution for your operating system <https://conda.io/miniconda.html>`_ (using the version for Python 3).

With Miniconda installed, you can create a new environment called ``"gisele"`` with all the necessary modules, by running the following command in a terminal or command-line window

  .. code-block:: fishshell

    $ conda env create -f environment.yml

To use Gisele, you need to activate the ``gisele`` environment each time

  .. code-block:: fishshell

    $ conda activate gisele

You are now ready to use Gisele together with the free and open source GLPK solver. However, we recommend to not use this solver where possible, since it performs relatively poorly (both in solution time and stability of result). Indeed, our example models use the free and open source CBC solver instead, but installing it on Windows requires an extra step. Read the next section for more information on installing alternative solvers.


.. _install_solvers:

Solvers
=======

You need at least one of the solvers supported by Pyomo installed. CBC (open-source) or Gurobi (commercial) are recommended for large problems, and have been confirmed to work with Gisele. Refer to the documentation of your solver on how to install it.

CBC
---

`CBC <https://projects.coin-or.org/Cbc>`_ is our recommended option if you want a free and open-source solver. CBC can be installed via conda on Linux and macOS by running ```conda install -c conda-forge coincbc```. Windows binary packages are somewhat more difficult to install, due to limited information on `the CBC website <https://projects.coin-or.org/Cbc>`_. We recommend you download the relevant binary for `CBC 2.9.9 <https://bintray.com/coin-or/download/Cbc/2.9.9>`_ and add `cbc.exe` to a directory known to PATH (e.g. an Anaconda environment 'bin' directory).

GLPK
____

