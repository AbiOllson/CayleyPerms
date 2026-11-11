"""This module contains the CayleyPermutation class and functions for working with them."""

from __future__ import annotations
from collections import deque
from functools import cached_property
from itertools import combinations
from typing import Any, Iterable, Iterator, Optional

DEBUG = False


class CayleyPermutation(tuple[int, ...]):
    """
    A Cayley Permutation is a list of integers with repeats allowed where
    if n is in the list, every k < n is in the list.

    Examples:
    >>> print(CayleyPermutation([0, 1, 2]))
    012
    >>> print(CayleyPermutation([1, 0, 2, 1]))
    1021
    """

    # pylint: disable=too-many-public-methods

    def __new__(cls, iterable: Iterable[int]):
        if DEBUG:
            iterable = tuple(iterable)
            if not (
                len(iterable) == 0
                or all(val in iterable for val in range(max(iterable) + 1))
            ):
                raise ValueError("Not a valid input to CayleyPermutation")
        return super(CayleyPermutation, cls).__new__(cls, iterable)

    def is_cayley_perm(self) -> bool:
        """Return True if input is correct"""
        return len(self) == 0 or all(val in self for val in range(max(self) + 1))

    def is_perm(self) -> bool:
        """Return True if the Cayley permutation is a permutation."""
        return len(set(self)) == len(self)

    def as_one_based(self) -> "str":
        """Returns Cayley permutation as a one based string from zero based.
        Adds parentheses for values greater than 9.

        Example:
        >>> CayleyPermutation([1, 2, 3, 0]).as_one_based()
        '2341'
        """
        return "".join(str(x + 1) if x < 9 else f"({x + 1})" for x in self)

    @classmethod
    def from_one_based(cls, cperm: Iterable[int]) -> "CayleyPermutation":
        """
        Return a Cayley permutation from a one based iterable.
        """
        return CayleyPermutation(tuple(x - 1 for x in cperm))

    def sub_cperms(self) -> set["CayleyPermutation"]:
        """Returns all sub-Cayley permutations of the Cayley permutation."""
        sub_cperms = set()
        next_cperms = set([self])
        while next_cperms:
            sub_cperms.update(next_cperms)
            next_cperms = set(
                sub_cperm
                for cperm in next_cperms
                for sub_cperm in cperm.remove_one_point()
            )
        return sub_cperms

    def remove_one_point(self) -> set["CayleyPermutation"]:
        """
        Returns all sub-Cayley permutations that are the Cayley permutation with one point removed.
        """
        sub_cperms: set[CayleyPermutation] = set()
        if len(self) == 0:
            return sub_cperms
        for i in range(len(self)):
            sub_cperms.add(self.standardise((self[:i] + self[i + 1 :])))
        return sub_cperms

    def is_simple(self) -> bool:
        """Returns true if the Cayley permutation is simple."""
        number_of_indices = len(self)
        for a, b in combinations(range(number_of_indices), 2):
            if len(self.interval(a, b)) < number_of_indices:
                return False
        return True

    def interval(self, idx1: int, idx2: int) -> list[int]:
        """
        Returns the smallest interval in the Cayley permutation
        that contains the indices idx1 and idx2.
        """
        indices_in_interval = list(range(idx1, idx2 + 1))
        new_indices_in_interval = self.add_to_interval(indices_in_interval)
        while new_indices_in_interval != list(
            range(min(new_indices_in_interval), max(new_indices_in_interval) + 1)
        ):
            indices_in_interval_no_gaps = list(
                range(min(new_indices_in_interval), max(new_indices_in_interval) + 1)
            )
            new_indices_in_interval = sorted(
                self.add_to_interval(indices_in_interval_no_gaps)
            )
        return new_indices_in_interval

    def add_to_interval(self, indices_in_interval: list[int]) -> list[int]:
        """For any values in the Cayley permutation that are in the range
        of the interval, adds their indices to the list of indices in the interval."""
        subcperm = [self[idx] for idx in indices_in_interval]
        max_val = max(subcperm)
        min_val = min(subcperm)
        values_in_interval = list(range(min_val, max_val + 1))
        for idx, val in enumerate(self):
            if idx not in indices_in_interval:
                if val in values_in_interval:
                    indices_in_interval.append(idx)
                    values_in_interval.append(val)
        return indices_in_interval

    def block_decomposition(self) -> list[list[int]]:
        """For a Cayley permutation, breaks it into intervals, begining with
        the leftmost largest interval and returns these as a list.

        Example:
        >>> CayleyPermutation([0, 1, 2, 1, 0]).block_decomposition()
        [[0], [1, 2, 3], [4]]
        """
        blocks = []
        current_index = 0
        while current_index < len(self):
            end_index = len(self) - 1
            while len(self.interval(current_index, end_index)) == len(self):
                if current_index == end_index:
                    break
                end_index -= 1
            if current_index == end_index:
                block = [current_index]
            else:
                block = self.interval(current_index, end_index)
            blocks.append(block)
            current_index = max(block) + 1
        return blocks

    def standardisation_of_block(self) -> "CayleyPermutation":
        """Returns the standardisation of the block of the Cayley permutation.
        (is the simple Cayley permutation that was inflated to give the Cayley permutation).
        """
        block_decomposition = self.block_decomposition()
        cperm = []
        for block in block_decomposition:
            cperm.append(self[block[0]])
        return CayleyPermutation.standardise(cperm)

    @classmethod
    def inflation(
        cls, simple_decomp: tuple["CayleyPermutation", tuple["CayleyPermutation", ...]]
    ) -> "CayleyPermutation":
        """Returns the inflation of the Cayley permutation from the simple decomposition.

        Example:
        >>> cperm = CayleyPermutation([0, 1, 2, 1, 0])
        >>> simple_decomp = cperm.simple_decomposition()
        >>> CayleyPermutation.inflation(simple_decomp)
        CayleyPermutation((0, 1, 2, 1, 0))
        """
        simple_cperm, blocks_decomp = simple_decomp
        simple = simple_cperm
        blocks = list(blocks_decomp)
        blocks_and_indices = []
        current_max = -1
        for i in range(max(simple) + 1):
            indices = []
            for idx, val in enumerate(simple):
                if val == i:
                    indices.append(idx)
            for idx in indices:
                new_block = [x + current_max + 1 for x in blocks[idx]]
                blocks_and_indices.append((new_block, idx))
            current_max = max(current_max, *new_block)
        cperm = []
        for i in range(len(blocks_and_indices)):
            cperm.extend(sorted(blocks_and_indices, key=lambda x: x[1])[i][0])
        return CayleyPermutation(cperm)

    def simple_decomposition(
        self,
    ) -> tuple["CayleyPermutation", tuple["CayleyPermutation", ...]]:
        """
        For a Cayley permutation, returns the tuple of the simple Cayley permutation
        it was inflated from and it's block decomposition.

        Example:
        >>> cperm = CayleyPermutation([0, 1, 2, 1, 0])
        >>> for p in cperm.simple_decomposition():
        ...     print(p)
        010
        (CayleyPermutation((0,)), CayleyPermutation((0, 1, 0)), CayleyPermutation((0,)))
        """
        blocks = self.block_decomposition()
        simple_cperm = self.standardisation_of_block()
        cperm_blocks = []
        for block in blocks:
            cperm_blocks.append(
                CayleyPermutation.standardise(self[block[0] : block[-1] + 1])
            )
        return simple_cperm, tuple(cperm_blocks)

    def sum_decomposable(self) -> bool:
        """Returns true if the Cayley permutation is sum decomposable."""
        for idx in range(len(self) - 1):
            interval = self.interval(0, idx)
            if len(interval) == len(self):
                return False
            for i in interval:
                if self[i] == 0:
                    return True
        return False

    def skew_decomposable(self) -> bool:
        """Returns true if the Cayley permutation is skew decomposable."""
        for idx in range(len(self) - 1, 0, -1):
            interval = self.interval(idx, len(self) - 1)
            if len(interval) == len(self):
                return False
            for i in interval:
                if self[i] == 0:
                    return True
        return False

    @classmethod
    def of_size(cls, size: int) -> list["CayleyPermutation"]:
        """
        Returns a list of all Cayley permutations of size 'size'.

        Examples:
        >>> CayleyPermutation.of_size(0)
        [CayleyPermutation(())]
        >>> CayleyPermutation.of_size(1)
        [CayleyPermutation((0,))]
        >>> sorted(CayleyPermutation.of_size(2))
        [CayleyPermutation((0, 0)), CayleyPermutation((0, 1)), CayleyPermutation((1, 0))]
        """
        cperms: list["CayleyPermutation"] = []
        if size == 0:
            return [CayleyPermutation([])]
        if size == 1:
            return [CayleyPermutation([0])]
        for cperm in CayleyPermutation.of_size(size - 1):
            cperms.extend(cperm.add_maximum())
        return cperms

    def insert(self, index, value):
        """Inserts value at index in the Cayley permutation."""
        return CayleyPermutation(self[:index] + [value] + self[index:])

    def subperm_from_indices(self, indices: Iterable[int]) -> "CayleyPermutation":
        """Returns the Cayley permutation at the indices."""
        return CayleyPermutation.standardise(self[idx] for idx in indices)

    def indices_above_value(self, value: int) -> list[int]:
        """Returns a list of the indices of the values that
        are greater than or equal to the input value."""
        above_max_indices = []
        for idx, val in enumerate(self):
            if val >= value:
                above_max_indices.append(idx)
        return above_max_indices

    def add_maximum(self) -> list["CayleyPermutation"]:
        """Adds a new maximum to the Cayley permutation in every possible way
        (one larger anywhere or the same as the current max at a smaller index).

        Example:
        >>> for cperm in CayleyPermutation([0, 1]).add_maximum():
        ...     print(cperm)
        201
        021
        012
        101
        011
        """
        val = max(self)
        index = self.index(val)
        perms = []
        for i in range(len(self) + 1):
            perms.append(CayleyPermutation(self[:i] + (val + 1,) + self[i:]))
        for i in range(index + 1):
            perms.append(CayleyPermutation(self[:i] + (val,) + self[i:]))
        return perms

    def contains(
        self, patterns: Iterable["CayleyPermutation"], require_last: int = 0
    ) -> bool:
        """
        Input a list of patterns and returns true if contains any of them.

        Searches only for patterns that must contain the last [require_last] entries.

        Examples:
        >>> CayleyPermutation([0, 1, 2]).contains([CayleyPermutation([0, 1])])
        True
        >>> CayleyPermutation([0, 1, 2]).contains([CayleyPermutation([0, 1]),
        ... CayleyPermutation([1, 0])])
        True
        >>> CayleyPermutation([0, 1, 2]).contains([CayleyPermutation([1, 0])])
        False
        """
        return any(self.contains_pattern(pattern, require_last) for pattern in patterns)

    def contains_pattern(
        self, pattern: "CayleyPermutation", require_last: int = 0
    ) -> bool:
        """
        Input one pattern and returns true if the pattern is contained.

        Searches only for patterns that must contain the last [require_last] entries.

        Examples:
        >>> CayleyPermutation([0, 1, 2]).contains_pattern(CayleyPermutation([0, 1]))
        True
        >>> CayleyPermutation([0, 1, 2]).contains_pattern(CayleyPermutation([1, 0]))
        False
        """
        if require_last:
            colours = [0] * (len(self) - require_last) + [1] * require_last
            patt_colours = [0] * (len(pattern) - require_last) + [1] * require_last
            return any(
                True for _ in pattern.occurrences_in(self, colours, patt_colours)
            )
        return any(True for _ in pattern.occurrences_in(self))

    def avoids(self, patterns: Iterable["CayleyPermutation"]) -> bool:
        """Returns true if the Cayley permutation avoids any of the patterns."""
        return not self.contains(patterns)

    def avoids_pattern(self, pattern: "CayleyPermutation") -> bool:
        """Returns true if the Cayley permutation avoids the pattern."""
        return not self.contains_pattern(pattern)

    @classmethod
    def standardise(cls, pattern: Iterable) -> "CayleyPermutation":
        """Returns the standardised version of a pattern.

        Example:
        >>> CayleyPermutation.standardise([2, 3])
        CayleyPermutation((0, 1))
        """
        pattern = tuple(pattern)
        key = sorted(set(pattern))
        stand = {}
        for i, v in enumerate(key):
            stand[v] = i
        return CayleyPermutation([stand[pat] for pat in pattern])

    def complement(self) -> "CayleyPermutation":
        """Returns the complement of the Cayley permutation."""
        n = max(self)
        return CayleyPermutation((n - x for x in self))

    def reverse(self) -> "CayleyPermutation":
        """Returns the reverse of the Cayley permutation."""
        return CayleyPermutation(self[::-1])

    def reverse_complement(self) -> "CayleyPermutation":
        """Returns the reverse complement of the Cayley permutation."""
        n = max(self)
        return CayleyPermutation((n - x for x in self[::-1]))

    def first_k_entries(self, k: int) -> list[int]:
        """Returns a list of the indices of the first k numbers
        that were inserted in the evolution of the Cayley permutation.

        Example:
        >>> CayleyPermutation([2, 0, 1, 2]).first_k_entries(2)
        [1, 2]
        >>> CayleyPermutation([0, 1, 0, 1, 2]).first_k_entries(3)
        [0, 2, 3]
        """
        current_min = 0
        indices: list[int] = []
        while len(indices) < k:
            mindices = []
            for idx, val in enumerate(self):
                if val == current_min:
                    mindices.append(idx)
            indices.extend(mindices[-(k - len(indices)) :])
            current_min += 1
        return sorted(indices)

    def last_k_entries(self, k: int) -> list[int]:
        """Returns a list of the indices of the last k numbers that were
        inserted in the evolution of the Cayley permutation.

        Example:
        >>> CayleyPermutation([2, 0, 1, 2]).last_k_entries(2)
        [0, 3]
        """
        current_max = max(self)
        indices: list[int] = []
        while len(indices) < k:
            maxindices = []
            for idx, val in enumerate(self):
                if val == current_max:
                    maxindices.append(idx)
            indices.extend(maxindices[: k - len(indices)])
            current_max -= 1
        return sorted(indices)

    def index_rightmost_max(self) -> int:
        """Returns the index of the rightmost maximum."""
        if len(self) == 0:
            # why?
            return 1
        max_val = max(self)
        for idx, val in reversed(list(enumerate(self))):
            if val == max_val:
                return idx
        raise ValueError("No maximum found.")

    def occurrences_in(
        self,
        word: tuple[int, ...],
        self_colours: Optional[Iterable[Any]] = None,
        patt_colours: Optional[Iterable[Any]] = None,
    ) -> Iterator[tuple[int, ...]]:
        """Find all indices of occurrences of self in word over natural numbers.
        If the optional colours are provided, in an occurrences the colours of
        the patterns have to match the colours of the permutation.

        Adapted from Ragnar Ardal's code in permuta.Perm.

        Examples:
        >>> list(CayleyPermutation((2, 0, 1)).occurrences_in(CayleyPermutation((5, 3, 0, 4, 2, 1))))
        [(0, 1, 3), (0, 2, 3), (0, 2, 4), (0, 2, 5), (1, 2, 4), (1, 2, 5)]
        >>> list(CayleyPermutation((1, 0)).occurrences_in(CayleyPermutation((1, 2, 3, 0))))
        [(0, 3), (1, 3), (2, 3)]
        >>> list(CayleyPermutation((0,)).occurrences_in(CayleyPermutation((1, 2, 3, 0))))
        [(0,), (1,), (2,), (3,)]
        >>> list(CayleyPermutation([]).occurrences_in(CayleyPermutation((1, 2, 3, 0))))
        [()]
        >>> list(CayleyPermutation((1, 0)).occurrences_in(CayleyPermutation((1, 2, 3, 0, 0))))
        [(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)]
        >>> list(CayleyPermutation((0, 0)).occurrences_in(CayleyPermutation((0, 1, 2, 1, 2))))
        [(1, 3), (2, 4)]

        """
        if not self:
            yield tuple()
            return
        if self_colours is not None:
            assert patt_colours is not None
            self_colours = tuple(self_colours)
            patt_colours = tuple(patt_colours)
            i = 0
            for cell in patt_colours:
                if self_colours[i] == cell:
                    i += 1
                if i == len(self_colours):
                    break
            else:
                # if we get here, we have not found all the colours, so no occurrence exists
                return

        if len(self) == 0:
            yield tuple()
            return
        if len(self) > len(word):
            return

        # The indices of the occurrence in perm.
        # This will be updated as we go through the possible occurrences.
        occurrence_indices = [0] * len(self)
        pattern_details = self._pattern_details
        number_of_values = max(word)

        def occurrences(i: int, k: int) -> Iterator[tuple[int, ...]]:
            # works with occurrences_indices and other defined variables
            # i is the index of the element in word being considered
            # k is how many elements of word that have already been added
            # to the occurrence
            elements_remaining = len(word) - i
            elements_needed = len(self) - k

            # lfi = left floor index
            # lci = left ceiling index
            # lbp = lower bound pre-computation
            # ubp = upper bound pre-computation
            (
                left_floor_idx,
                left_ceil_idx,
                lower_bound_points,
                upper_bound_points,
            ) = pattern_details[k]
            if left_floor_idx == -1:
                # no left index so element of occurrence must be at least self[k]
                lower_bound = lower_bound_points
            else:
                # new element of occurrence must be at least as from its
                # left floor as self[k] is from its left floor
                lower_bound = (
                    word[occurrence_indices[left_floor_idx]] + lower_bound_points
                )
            if left_ceil_idx == -1:
                # no left ceil index fo new element must be at least less
                # than its left ceiling as self[k] is to its left ceiling
                upper_bound = number_of_values - upper_bound_points
            else:
                # new element of occurrence must at least less than its
                # left ceiling as self[k] is to its left ceiling
                upper_bound = (
                    word[occurrence_indices[left_ceil_idx]] - upper_bound_points
                )

            # Loop over remaining elements of perm / the index i
            while True:
                if elements_remaining < elements_needed:
                    # can't form an occurrence as not enough points
                    return
                element = word[i]
                if (
                    self_colours is None
                    or patt_colours[i] == self_colours[k]  # type: ignore[index]
                ):
                    if lower_bound <= element <= upper_bound:
                        occurrence_indices[k] = i
                        if elements_needed == 1:
                            yield tuple(occurrence_indices)
                        else:
                            yield from occurrences(i + 1, k + 1)
                i += 1
                elements_remaining -= 1

        yield from occurrences(0, 0)

    @cached_property
    def number_of_values(self):
        """
        Return the number of unique values in the Cayley permutation.
        """
        return max(self)

    @cached_property
    def _pattern_details(self) -> list[tuple[int, int, int, int]]:
        """For each point (i, pi(i)) in self, return the tuple (j, k, l, m)
        where i is index of the floor, k is the index of the ceiling, l is
        the difference between pi(i) and pi(j), and m is the difference
        between pi(i) and pi(j).

        floor:      index j < i of the greatest value such that pi(j) <= pi(i)
        ceiling:    index j < i of the least value such that pi(j) >= pi(i

        Note, floor and ceiling default to -1 if does not exist.

        Adapted from Ragnar Ardal's code in permuta.Perm.
        """
        return [
            (
                floor,
                ceiling,
                val if floor == -1 else val - self[floor],
                self.number_of_values - val if ceiling == -1 else self[ceiling] - val,
            )
            for val, (floor, ceiling) in zip(self, self._left_floor_and_ceiling())
        ]

    def _left_floor_and_ceiling(self) -> Iterator[tuple[int, int]]:
        """For each idx, val pair in the perm yield the value together
        the left floor index and left floor ceiling.

        Adapted from Ragnar Ardal's code in permuta.Perm.
        """
        # deq will keep small values to left and big values to right
        deq: deque[tuple[int, int]] = deque()
        smallest, biggest = -1, -1
        for idx, val in enumerate(self):
            # iterate from left to right
            if idx == 0:
                deq.append((idx, val))
                smallest, biggest = val, val
                yield (-1, -1)
            elif val < smallest:
                # new smallest value, so no left floor
                while deq[0][1] != smallest:
                    # rotate until smallest val is at front
                    deq.rotate(-1)
                yield (-1, deq[0][0])
                deq.appendleft((idx, val))
                smallest = val
            elif val > biggest:
                # new largest value so no left ceiling
                while deq[-1][1] != biggest:
                    deq.rotate(-1)
                yield (deq[-1][0], -1)
                deq.append((idx, val))
                biggest = val
            else:
                while not deq[-1][1] <= val <= deq[0][1]:
                    deq.rotate(1)
                if val == deq[-1][1]:
                    yield (deq[-1][0], deq[-1][0])
                elif val == deq[0][1]:
                    yield (deq[0][0], deq[0][0])
                else:
                    yield (deq[-1][0], deq[0][0])
                deq.appendleft((idx, val))

    def occurrences(
        self, basis: Iterable["CayleyPermutation"]
    ) -> dict["CayleyPermutation", tuple[tuple[int, ...], ...]]:
        """Returns a dictionary of the occurrences of a pattern in the basis
        and indices of the Cayley permutation where they occur.

        Example:
        >>> basis = [CayleyPermutation([0, 0])]
        >>> CayleyPermutation([0, 1, 2, 1, 2]).occurrences(basis)
        {CayleyPermutation((0, 0)): ((1, 3), (2, 4))}
        """
        return {pattern: tuple(pattern.occurrences_in(self)) for pattern in basis}

    def avoids_same_after_deleting(
        self, basis: Iterable["CayleyPermutation"], index: int
    ) -> bool:
        """
        Returns true if the Cayley permutation avoids
        the basis still after deleting the index.
        """
        basis = tuple(basis)
        if self.contains(basis):
            cperm_deleted = self.delete_index(index)
            if not cperm_deleted.contains(basis):
                return False
        return True

    def delete_index(self, index: int) -> "CayleyPermutation":
        """Returns a Cayley permutation with the index deleted."""
        return CayleyPermutation.standardise(self[:index] + self[index + 1 :])

    def is_monotonically_decreasing(self) -> bool:
        """Returns true if the Cayley permutation is monotonicaly decreasing.

        Example:
        >>> CayleyPermutation([2, 1, 0, 0]).is_monotonically_decreasing()
        True
        """
        for first, second in zip(self[:-1], self[1:]):
            if first < second:
                return False
        return True

    def is_monotonically_increasing(self) -> bool:
        """Returns true if the Cayley permutation is monotonicaly increasing.

        Example:
        >>> CayleyPermutation([0, 1, 2, 2]).is_monotonically_increasing()
        True
        """
        for first, second in zip(self[:-1], self[1:]):
            if first > second:
                return False
        return True

    def is_increasing(self) -> bool:
        """Returns true if the Cayley permutation is strictly increasing.

        Example:
        >>> CayleyPermutation([0, 1, 2, 2]).is_increasing()
        False
        >>> CayleyPermutation([0, 1, 2]).is_increasing()
        True
        """
        for first, second in zip(self[:-1], self[1:]):
            if first >= second:
                return False
        return True

    def is_decreasing(self) -> bool:
        """Returns true if the Cayley permutation is strictly decreasing.

        Example:
        >>> CayleyPermutation([2, 1, 0, 0]).is_decreasing()
        False
        >>> CayleyPermutation([2, 1, 0]).is_decreasing()
        True
        """
        for first, second in zip(self[:-1], self[1:]):
            if first <= second:
                return False
        return True

    def is_constant(self) -> bool:
        """Returns true if the Cayley permutation is constant.

        Example:
        >>> CayleyPermutation([0, 0, 1, 0]).is_constant()
        False
        """
        for first, second in zip(self[:-1], self[1:]):
            if first != second:
                return False
        return True

    def check_is_strict(self) -> bool:
        """
        Returns true if the Cayley permutation is strictly increasing,
        strictly decreasing, or constant.
        """
        if self.is_increasing():
            return True
        if self.is_decreasing():
            return True
        if self.is_constant():
            return True
        return False

    def is_rgf(self) -> bool:
        """Returns true if the Cayley permutation is an RGF.
        To be an RGF, any number in the Cayley permutation must have
        every number smaller than it at smaller indices.

        Example:
        >>> CayleyPermutation([0, 1, 2, 1, 0]).is_rgf()
        True
        >>> CayleyPermutation([1, 2, 1, 0]).is_rgf()
        False
        """
        if len(self) == 0 or len(self) == 1:
            return True
        max_val = max(self)
        for i in range(max_val + 1):
            for val in self:
                if val > i:
                    return False
                if val == i:
                    break
        return True

    def is_rgf_of_matching(self) -> bool:
        """Returns true if the Cayley permutations is an RGF
        of a matching - if every numjber appears exactly twice."""
        if len(self) == 0:
            return True
        if not self.is_rgf():
            return False
        for i in range(max(self) + 1):
            if self.count(i) != 2:
                return False
        return True

    def as_canonical(self) -> Iterator["CayleyPermutation"]:
        """Converts a Cayley permutation into a list
        of Cayley permutations that are in canonical form.
        A state is a 4-tuple of a Cayley perm and indices

        Examples:
        >>> list(CayleyPermutation([2, 1, 0]).as_canonical())
        [CayleyPermutation((0, 1, 2, 1, 0))]
        >>> list(CayleyPermutation([0, 1, 0, 3, 2]).as_canonical())
        [CayleyPermutation((0, 1, 0, 2, 3, 2)), CayleyPermutation((0, 1, 2, 0, 3, 2))]
        """
        idx_current_max, val_current_max, working_index = -1, -1, 0
        states: list[tuple[tuple[int, ...], int, int, int]] = [
            (self, idx_current_max, val_current_max, working_index)
        ]
        while states:
            new_states: list[tuple[tuple[int, ...], int, int, int]] = []
            for state in states:
                if len(state[0]) == state[3]:
                    yield CayleyPermutation(state[0])
                else:
                    new_states.extend(self._fix_first_max(state))
            states = new_states

    def _fix_first_max(
        self, state: tuple[tuple[int, ...], int, int, int]
    ) -> Iterator[tuple[tuple[int, ...], int, int, int]]:
        """Checks values in a Cayley permutation to see if they are in canonical form
        and if not then inserts the numbers needed in every possible way"""
        cperm, idx_current_max, val_current_max, working_index = state
        if cperm[working_index] <= val_current_max:
            working_index += 1
            new_state = (cperm, idx_current_max, val_current_max, working_index)
            yield new_state
        elif cperm[working_index] == val_current_max + 1:
            val_current_max += 1
            idx_current_max = working_index
            working_index += 1
            new_state = (cperm, idx_current_max, val_current_max, working_index)
            yield new_state
        elif cperm[working_index] > val_current_max + 1:
            list_a = cperm[idx_current_max + 1 : working_index]
            list_b = tuple(range(val_current_max + 1, cperm[working_index]))
            new_val_current_max = cperm[working_index]
            new_idx_current_max = working_index + len(list_b)
            new_working_index = working_index + 1 + len(list_b)
            for shuff in self.shuffle(list_a, list_b):
                new_cperm = (
                    cperm[: idx_current_max + 1] + tuple(shuff) + cperm[working_index:]
                )
                new_state = (
                    new_cperm,
                    new_idx_current_max,
                    new_val_current_max,
                    new_working_index,
                )
                yield new_state

    @staticmethod
    def shuffle(
        list_a: tuple[int, ...], list_b: tuple[int, ...]
    ) -> Iterator[tuple[int, ...]]:
        """Returns all possible shuffles of two lists list_a and list_b.

        Example:
        >>> for shuff in CayleyPermutation.shuffle([1, 2], [3, 4]):
        ...     print(shuff)
        (1, 2, 3, 4)
        (1, 3, 2, 4)
        (1, 3, 4, 2)
        (3, 1, 2, 4)
        (3, 1, 4, 2)
        (3, 4, 1, 2)
        """
        length_a = len(list_a)
        length_b = len(list_b)
        for a_indices in combinations(range(length_a + length_b), length_a):
            b_indices = [i for i in range(length_a + length_b) if i not in a_indices]
            shuff = list(range(length_a + length_b))
            for idx_a, idx_shuff in enumerate(a_indices):
                shuff[idx_shuff] = list_a[idx_a]
            for idx_b, idx_shuff in enumerate(b_indices):
                shuff[idx_shuff] = list_b[idx_b]
            yield tuple(shuff)

    def old_ascii_plot(self) -> str:
        """Returns an ascii plot of the mesh pattern."""
        if len(self) == 0:
            return "+---+\n|   |\n+---+\n"
        n = len(self)
        m = max(self)
        rows = []
        for i in range(m + 1):
            new_row = []
            for j in self:
                if j == i:
                    new_row.append(" \u25cf ")
                else:
                    new_row.append("   ")
            rows.append(new_row)
        rows_as_strings = []
        for row in rows:
            rows_as_strings.append("|" + "|".join(row) + "|" + "\n")
        edge_row = "---".join("+" for _ in range(n + 1)) + "\n"
        grid = edge_row + edge_row.join(reversed(rows_as_strings)) + edge_row
        return grid

    def ascii_plot(self) -> str:
        """Returns an ascii plot of the Cayley permutation."""
        if len(self) == 0:
            return "+---+\n|   |\n+---+\n"
        n = len(self)
        m = max(self)
        empty_cell = "   "
        point = "\u25cf"
        normal_row = "---"
        crossing_lines = "+"
        normal_column = "|"
        point_rows = []
        for i in range(m + 1):
            new_row = normal_row
            for j in self:
                if j == i:
                    new_row += point + normal_row
                else:
                    new_row += crossing_lines + normal_row
            new_row += "\n"
            point_rows.append(new_row)
        empty_row = normal_column.join(empty_cell for _ in range(n + 1)) + "\n"
        grid = empty_row + empty_row.join(reversed(point_rows)) + empty_row
        return grid

    def gridding_ascii_plot(self, gridding) -> str:
        """Returns an ascii plot of the 2x2 gridding."""
        top, bottom = gridding
        top_left, top_right = top
        bottom_left, bottom_right = bottom

        if len(self) == 0:
            return "+---+\n|   |\n+---+\n"
        normal_row = "---"
        point_rows = []
        for i in range(self.number_of_values + 1):
            new_row = normal_row
            for j in self:
                if j == i:
                    if j in top_left:
                        new_row += "🔵" + normal_row
                    elif j in top_right:
                        new_row += "🔴" + normal_row
                    elif j in bottom_left:
                        new_row += "🟣" + normal_row
                    elif j in bottom_right:
                        new_row += "🟢" + normal_row
                else:
                    new_row += "+ " + normal_row
            new_row += "\n"
            point_rows.append(new_row)
        empty_row = "| ".join("   " for _ in range(len(self) + 1)) + "\n"
        grid = empty_row + empty_row.join(reversed(point_rows)) + empty_row
        return grid

    def to_jsonable(self) -> dict:
        """Returns a dictionary of the Cayley permutation."""
        return {"cperm": self}

    @classmethod
    def from_dict(cls, d: dict) -> "CayleyPermutation":
        """Returns a Cayley permutation from a dictionary."""
        return cls(d["cperm"])

    def __str__(self):
        if len(self) == 0:
            return "\u03b5"
        return "".join(str(x) if x < 10 else f"({str(x)})" for x in self)

    def __repr__(self):
        return f"CayleyPermutation({tuple(self)})"

    def __lt__(self, other) -> bool:
        return (len(self), tuple(self)) < (len(other), tuple(other))

    def __le__(self, other) -> bool:
        return (len(self), tuple(self)) <= (len(other), tuple(other))
