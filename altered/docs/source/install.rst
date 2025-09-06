Installation guide
==================

How do I set it up?
^^^^^^^^^^^^^^^^^^^

You can install MadraX either from the bitbucket (https://bitbucket.org/grogdrinker/madrax/) repository, from a pypi package (https://pypi.org/project/madrax/) or from a conda package (https://anaconda.org/anaconda/madrax).


Installing MadraX with pip:
^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to install MadraX via pypi, we suggest to install pytorch separately following the instructions from https://pytorch.org/get-started/locally/

Then just open a terminal and type:

``python -m pip install "madrax @git+https://bitbucket.org/grogdrinker/madrax/"``

or, if you prefer, simply 

``pip install "madrax @git+https://bitbucket.org/grogdrinker/madrax/"``

and all the required dependencies will be installed automatically. Congrats, you are ready to rock!

If something goes wrong with pytorch installation, we suggest to install it separately following the instructions present in https://pytorch.org/


Installing MadraX with Anaconda (Linux Only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MadraX can be installed simply via conda by typing in the terminal:

``conda install -c grogdrinker madrax``

again, all the required dependencies will be installed automatically.

