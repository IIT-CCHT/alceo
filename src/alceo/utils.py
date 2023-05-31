"""A set of utility functions used throughout the project.
"""

def in_notebook() -> bool:
    """Utility to check if the code is running in a IPKernel (like Jupyter or IPython cells).
    
    Code taken from: https://stackoverflow.com/a/22424821

    Returns:
        bool: is the code running in a notebook?
    """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

