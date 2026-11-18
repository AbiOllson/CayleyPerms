"""This module is used for checking if a sequence
is a vertical juxtaposition and if a basis has a regular
vertical insertion encoding."""

from cayley_permutations import string_to_basis, CayleyPermutation


# pylint: disable=duplicate-code
def regular_vertical_insertion_encoding(
    basis: str | tuple[CayleyPermutation, ...],
) -> bool:
    """Checks if a basis has a regular insertion encoding.

    Example:
    >>> regular_vertical_insertion_encoding("01_10")
    True
    """
    basis = string_to_basis(basis) if isinstance(basis, str) else basis
    for i in range(3):
        for j in range(3):
            if any(checks_vert_type(list(cperm), (i, j)) for cperm in basis):
                continue
            return False
    return True


def checks_vert_type(cperm: list[int], class_to_check: tuple[int, int]) -> bool:
    """
    Returns True if the sequence is a vertical juxtaposition
    of the type specified by the tuple.
    In the tuple the first element is above, the second element is below.
    0 -> strictly decreasing
    1 -> strictly increasing
    2 -> constant

    Examples:
    >>> checks_vert_type([0, 1, 2], (1, 1))
    True
    >>> checks_vert_type([0, 1, 2], (0, 0))
    False
    """
    if len(cperm) == 0 or len(cperm) == 1:
        return True
    if class_to_check[0] == 2:
        return check_con_bottom(cperm, class_to_check[1])
    if class_to_check[1] == 2:
        return check_con_top(cperm, class_to_check[0])
    if class_to_check[1] == 1:
        return check_inc_bottom(cperm, class_to_check[0])
    if class_to_check[1] == 0:
        return check_dec_bottom(cperm, class_to_check[0])
    raise ValueError("Invalid class_to_check value. Must be 0, 1, or 2.")


def check_con_bottom(cperm: list[int], class_to_check: int) -> bool:
    """Checks for if the sequence is constant on bottom"""
    if class_to_check == 2:
        return con_con(cperm)
    if class_to_check == 0:
        return dec_con(cperm)
    if class_to_check == 1:
        return inc_con(cperm)
    raise ValueError("Invalid class_to_check value. Must be 0, 1, or 2.")


def check_con_top(cperm: list[int], class_to_check: int) -> bool:
    """Checks for if the sequence is constant on top"""
    if class_to_check == 0:
        return con_dec(cperm)
    if class_to_check == 1:
        return con_inc(cperm)
    raise ValueError("Invalid class_to_check value. Must be 0, 1, or 2.")


def check_inc_bottom(cperm: list[int], class_to_check: int) -> bool:
    """Checks for if the sequence is increasing on bottom"""
    if class_to_check == 0:
        return inc_dec(cperm)
    if class_to_check == 1:
        return inc_inc(cperm)
    raise ValueError("Invalid class_to_check value. Must be 0, 1, or 2.")


def check_dec_bottom(cperm: list[int], class_to_check: int) -> bool:
    """Checks for if the sequence is decreasing on bottom"""
    if class_to_check == 1:
        return dec_inc(cperm)
    if class_to_check == 0:
        return dec_dec(cperm)
    raise ValueError("Invalid class_to_check value. Must be 0, 1, or 2.")


def inc_inc(cperm: list[int], min_height: int = 0) -> bool:
    """Returns True if the sequence is increasing on bottom
    and increasing on top.
    NOTE: input list must be a Cayley permutation.

    Creates bottom sequence with the first values in cperm
    that are at the same index as their value. The first value
    that is not at the same index is added to the top sequence
    and a line is drawn under it. Any other value must form an
    increasing sequence above or below that line.
    """
    top_seq = []
    bottom_seq = []
    for idx, val in enumerate(cperm):
        if val == idx:
            bottom_seq.append(val)
        else:
            top_seq.append(val)
            if val <= min_height:
                return False
            if bottom_seq and val <= bottom_seq[-1]:
                return False
            break
    if not top_seq:
        return True
    line = top_seq[0]
    start_index = len(bottom_seq) + 1

    for val in cperm[start_index:]:
        if val < line:
            if not bottom_seq:
                bottom_seq.append(val)
            elif val <= bottom_seq[-1]:
                return False
            bottom_seq.append(val)
        else:
            if val <= top_seq[-1]:
                return False
            top_seq.append(val)
    return True


def dec_dec(cperm: list[int]) -> bool:
    """Returns True if the sequence is decreasing on bottom
    and decreasing on top.
    NOTE: input list must be a Cayley permutation.

    Does the same as inc_inc, but in reverse order.
    """
    reversed_cperm = cperm[::-1]
    return inc_inc(reversed_cperm)


def inc_dec(cperm: list[int]) -> bool:
    """Returns True if the sequence is increasing on bottom
    and decreasing on top."""
    middle_val = cperm[-1]
    last_val_top = middle_val
    last_val_bottom = middle_val
    for val in cperm[-2::-1]:
        if val < middle_val:
            if val >= last_val_bottom:
                return False
            last_val_bottom = val
        else:
            if val <= last_val_top:
                return False
            last_val_top = val
    return True


def dec_inc(cperm: list[int], min_height: int = 0) -> bool:
    """Returns True if the sequence is decreasing on bottom
    and increasing on top."""
    middle_val = cperm[0]
    last_val_top = middle_val
    last_val_bottom = middle_val
    for val in cperm[1:]:
        if val < middle_val:
            if val >= last_val_bottom:
                return False
            last_val_bottom = val
        else:
            if val <= min_height:
                return False
            if val <= last_val_top:
                return False
            last_val_top = val
    return True


def con_con(cperm: list[int]) -> bool:
    """Returns True if the sequence is constant on bottom
    and constant on top."""
    max_val = max(cperm)
    if max_val == 0:
        return True
    for val in cperm:
        if val not in {max_val, 0}:
            return False
    return True


def con_inc(cperm: list[int]) -> bool:
    """Returns True if the sequence is constant on bottom
    and increasing on top."""
    initial_val = 0
    for val in cperm:
        if val != 0:
            if val <= initial_val:
                return False
            initial_val = val
    return True


def con_dec(cperm: list[int]) -> bool:
    """Returns True if the sequence is constant on bottom
    and decreasing on top.
    Looks at vals in reverse order."""
    initial_val = 0
    for val in cperm[-1::-1]:
        if val != 0:
            if val <= initial_val:
                return False
            initial_val = val
    return True


def inc_con(cperm: list[int]) -> bool:
    """Returns True if the sequence is increasing on bottom
    and constant on top."""
    max_val = max(cperm)
    initial_val = max_val
    for val in cperm[-1::-1]:
        if val != max_val:
            if val >= initial_val:
                return False
            initial_val = val
    return True


def dec_con(cperm: list[int]) -> bool:
    """Returns True if the sequence is decreasing on bottom
    and constant on top."""
    max_val = max(cperm)
    initial_val = max_val
    for val in cperm:
        if val != max_val:
            if val >= initial_val:
                return False
            initial_val = val
    return True
