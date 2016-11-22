"""
Common objects and methods related to data typing
"""


def is_categorical(col):
    """
    Returns True if pandas series contains categorical data

    A column is considered categorical if:
    1) The series does not have a numeric data type, or
    2) The series is a dummy variable, ie contains only {0, 1, NaN}, or
    """
    return (not is_numeric(col)) or (is_dummy_variable(col))


def is_dummy_variable(col):
    """
    Returns True if a pandas series is likely a dummy variable

    A dummy variable has a non-null cardinality of 2
    """
    return len(col.dropna().unique()) == 2


def is_numeric(col):
    """
    Returns True of a pandas series has a numeric data type
    """
    numerictypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    return col.dtypes in numerictypes