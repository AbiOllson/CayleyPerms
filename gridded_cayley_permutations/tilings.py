"""
The Tiling class, that inherits from CombinatorialClass.

A tiling represents the set of gridded Cayley permutations with cells coming up to a given
dimension, that avoid a set of obstructions and contain a set of requirements.
"""

from collections import defaultdict
from functools import cached_property
from itertools import chain, product, combinations, combinations_with_replacement
from math import factorial
from typing import Iterable, Iterator, Optional

from comb_spec_searcher import CombinatorialClass

from cayley_permutations import CayleyPermutation, string_to_basis
from check_regular_ins_enc import (
    regular_vertical_insertion_encoding,
    regular_horizontal_insertion_encoding,
)

from .gridded_cayley_perms import GriddedCayleyPerm
from .minimal_gridded_cperms import MinimalGriddedCayleyPerm
from .row_col_map import RowColMap
from .simplify_obstructions_and_requirements import SimplifyObstructionsAndRequirements

Cell = tuple[int, int]


def binomial(x: int, y: int) -> int:
    """Return the binomial coefficient x choose y."""
    try:
        return factorial(x) // factorial(y) // factorial(x - y)
    except ValueError:
        return 0


# pylint: disable=too-many-lines


class Tiling(CombinatorialClass):
    """A tiling represents the set of gridded Cayley permutations with cells coming up to a given
    dimension, that avoid a set of obstructions and contain a set of requirements."""

    # pylint: disable=too-many-public-methods
    def __init__(
        self,
        obstructions: Iterable[GriddedCayleyPerm],
        requirements: Iterable[Iterable[GriddedCayleyPerm]],
        dimensions: tuple[int, int],
        simplify=True,
    ) -> None:
        self.obstructions = tuple(obstructions)
        self.requirements = tuple(tuple(req) for req in requirements)
        self.dimensions = (dimensions[0], dimensions[1])

        algorithm = SimplifyObstructionsAndRequirements(
            self.obstructions, self.requirements, self.dimensions
        )
        if simplify:
            algorithm.simplify()
        self.obstructions = algorithm.obstructions
        self.requirements = algorithm.requirements

    def simplify(self) -> "Tiling":
        """Simplifies the tiling and returns a new tiling."""
        algorithm = SimplifyObstructionsAndRequirements(
            self.obstructions, self.requirements, self.dimensions
        )
        algorithm.simplify()
        return Tiling(algorithm.obstructions, algorithm.requirements, self.dimensions)

    def _gridded_cayley_permutations(self, size: int) -> Iterator[GriddedCayleyPerm]:
        """
        Generating gridded Cayley permutations of size 'size'.
        """
        if size == 0:
            if not GriddedCayleyPerm(CayleyPermutation([]), []) in self.obstructions:
                yield GriddedCayleyPerm(CayleyPermutation([]), [])
            return
        for gcp in self._gridded_cayley_permutations(size - 1):
            next_ins = gcp.next_insertions(self.dimensions)
            for val, cell in next_ins:
                next_gcp = gcp.insertion_different_value(val, cell)
                if self.satisfies_obstructions(next_gcp):
                    yield next_gcp
                if val in gcp.pattern:
                    if cell[1] == gcp.row_containing_value(val):
                        next_gcp = gcp.insertion_same_value(val, cell)
                        if self.satisfies_obstructions(next_gcp):
                            yield next_gcp

    def gridded_cayley_permutations(self, size: int) -> Iterator[GriddedCayleyPerm]:
        """Generating gridded Cayley permutations of size 'size' (that satisfy the requirements)."""
        yield from filter(
            self.satisfies_requirements, self._gridded_cayley_permutations(size)
        )

    def satisfies_obstructions(self, gcp: GriddedCayleyPerm) -> bool:
        """
        Checks whether a single gridded Cayley permutation satisfies the obstructions.
        """
        return not gcp.contains(self.obstructions)

    def satisfies_requirements(self, gcp: GriddedCayleyPerm) -> bool:
        """
        Checks whether a single gridded Cayley permutation satisfies the requirements.
        """
        for req in self.requirements:
            if not gcp.contains(req):
                return False
        return True

    def gcp_in_tiling(self, gcp: GriddedCayleyPerm) -> bool:
        """
        Checks whether a single gridded Cayley permutation is in the tiling.
        """
        return (
            all(
                x < self.dimensions[0] and y < self.dimensions[1]
                for x, y in gcp.positions
            )
            and self.satisfies_obstructions(gcp)
            and self.satisfies_requirements(gcp)
        )

    @cached_property
    def active_cells(self) -> set[tuple[int, int]]:
        """Returns the set of active cells in the tiling.
        (Cells are active if they do not contain a point obstruction.)"""
        return SimplifyObstructionsAndRequirements(
            self.obstructions, self.requirements, self.dimensions
        ).active_cells()

    def positive_cells(self) -> set[tuple[int, int]]:
        """Returns a set of cells that are positive in the tiling.
        (Cells are positive if they contain a point requirement.)"""
        positive_cells = set()
        for req_list in self.requirements:
            current = set(req_list[0].positions)
            for req in req_list:
                current = current.intersection(req.positions)
            positive_cells.update(current)
        return positive_cells

    def point_cells(self) -> set[tuple[int, int]]:
        """Returns the set of cells that can only contain a point."""
        # can this be made more efficient?
        point_cells = set()
        for cell in self.positive_cells():
            if (
                GriddedCayleyPerm(CayleyPermutation([0, 1]), [cell, cell])
                in self.obstructions
                and GriddedCayleyPerm(CayleyPermutation([1, 0]), [cell, cell])
                in self.obstructions
                and GriddedCayleyPerm(CayleyPermutation([0, 0]), [cell, cell])
                in self.obstructions
            ):
                point_cells.add(cell)
        return point_cells

    def not_blank_cells(self) -> set[tuple[int, int]]:
        """Returns the set of cells that are a position for some ob or req."""
        if not self.obstructions and not self.requirements:
            return set()
        combined_obs_and_reqs = self.obstructions + tuple(chain(*self.requirements))
        return set(chain(*(gcp.positions for gcp in combined_obs_and_reqs)))

    def blank_cells(self) -> set[tuple[int, int]]:
        """Returns the set of cells that contain no obs or reqs."""
        return (
            set(product(range(self.dimensions[0]), range(self.dimensions[1])))
            - self.not_blank_cells()
        )

    def empty_cells(self) -> set[Cell]:
        """Returns the set of empty cells"""
        empty = set()
        for ob in self.obstructions:
            if len(ob) == 1:
                empty.add(ob.positions[0])
        return empty

    def single_value_cells(self) -> set[Cell]:
        """Returns the set of cells with at most one value"""
        cells = set[Cell]()
        for cell in self.active_cells:
            if (
                GriddedCayleyPerm((0, 1), (cell, cell)) in self.obstructions
                and GriddedCayleyPerm((1, 0), (cell, cell)) in self.obstructions
            ):
                cells.add(cell)
        return cells

    def single_position_cells(self) -> set[Cell]:
        """Returns the set of cells with at most one position"""
        cells = set[Cell]()
        for cell in self.active_cells:
            if (
                GriddedCayleyPerm((0, 1), (cell, cell)) in self.obstructions
                and GriddedCayleyPerm((1, 0), (cell, cell)) in self.obstructions
                and GriddedCayleyPerm((0, 0), (cell, cell)) in self.obstructions
            ):
                cells.add(cell)
        return cells

    def requirement_cells(self) -> set[Cell]:
        """Returns every cell that contains a requirement"""
        if not self.requirements:
            return set[Cell]()
        return set(chain(*(set(req.positions) for req in chain(*self.requirements))))

    def obs_by_col_and_row(
        self,
    ) -> tuple[dict[int, set[GriddedCayleyPerm]], dict[int, set[GriddedCayleyPerm]]]:
        """Returns a dict with obstructions sorted by intersecting columns
        and a dict with obstructions sorted by intersecting rows"""
        by_col = defaultdict[int, set[GriddedCayleyPerm]](set)
        by_row = defaultdict[int, set[GriddedCayleyPerm]](set)
        for ob in self.obstructions:
            for pos in set(ob.positions):
                by_col[pos[0]].add(ob)
                by_row[pos[1]].add(ob)
        return by_col, by_row

    def reqs_by_col_and_row(
        self,
    ) -> tuple[
        dict[int, set[tuple[GriddedCayleyPerm, ...]]],
        dict[int, set[tuple[GriddedCayleyPerm, ...]]],
    ]:
        """Returns a dict with requirements sorted by intersecting columns
        and a dict with requirements sorted by intersecting rows"""
        by_col = defaultdict[int, set[tuple[GriddedCayleyPerm, ...]]](set)
        by_row = defaultdict[int, set[tuple[GriddedCayleyPerm, ...]]](set)
        for req_list in self.requirements:
            for pos in set(chain.from_iterable(req.positions for req in req_list)):
                by_col[pos[0]].add(req_list)
                by_row[pos[1]].add(req_list)
        return by_col, by_row

    def delete_columns(self, cols: Iterable[int]) -> "Tiling":
        """
        Deletes columns at indices specified
        from the tiling and returns the new tiling.
        """
        return self.delete_rows_and_columns(cols, [])

    def delete_rows(self, rows: Iterable[int]) -> "Tiling":
        """
        Deletes rows at indices specified
        from the tiling and returns the new tiling.
        """
        return self.delete_rows_and_columns([], rows)

    def delete_rows_and_columns(
        self, cols: Iterable[int], rows: Iterable[int]
    ) -> "Tiling":
        """
        Deletes rows and columns at indices specified
        from the tiling and returns the new tiling.
        """
        return self.tiling_and_rc_map_after_deleting_rows_and_columns(cols, rows)[0]

    def tiling_and_rc_map_after_deleting_rows_and_columns(
        self, cols: Iterable[int], rows: Iterable[int]
    ) -> tuple["Tiling", RowColMap]:
        """
        Deletes rows and columns at indices specified
        from the tiling and returns the new tiling
        and the row column map used to create it.
        """
        rows, cols = set(rows), set(cols)
        col_map = {}
        counter = 0
        for ind in range(self.dimensions[0]):
            if ind in cols:
                continue
            col_map[ind] = counter
            counter += 1

        row_map = {}
        counter = 0
        for ind in range(self.dimensions[1]):
            if ind in rows:
                continue
            row_map[ind] = counter
            counter += 1
        rc_map = RowColMap(col_map, row_map)
        new_obstructions = tuple(
            ob
            for ob in self.obstructions
            if all(x not in cols and y not in rows for x, y in ob.positions)
        )

        new_obstructions = rc_map.map_gridded_cperms(new_obstructions)
        new_requirements = []
        for req_list in self.requirements:
            new_req_list = tuple(
                req
                for req in req_list
                if all(x not in cols and y not in rows for x, y in req.positions)
            )
            if new_req_list:
                new_requirements.append(new_req_list)
        new_requirements = list(rc_map.map_requirements(new_requirements))
        new_dimensions = (
            self.dimensions[0] - len(cols),
            self.dimensions[1] - len(rows),
        )
        new_til = Tiling(
            new_obstructions, new_requirements, new_dimensions, simplify=False
        )
        return new_til, rc_map

    def find_empty_rows_and_columns(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Returns a list of the indices of empty rows and
        a list of the indices of empty columns."""
        if self.dimensions == (0, 0):
            return tuple(), tuple()
        col_count: dict[int, int] = defaultdict(int)
        row_count: dict[int, int] = defaultdict(int)
        for ob in self.obstructions:
            if len(ob) == 1:
                col_count[ob.positions[0][0]] += 1
                row_count[ob.positions[0][1]] += 1
        empty_cols = tuple(
            col for col, count in col_count.items() if count == self.dimensions[1]
        )
        empty_rows = tuple(
            row for row, count in row_count.items() if count == self.dimensions[0]
        )
        return empty_cols, empty_rows

    def requirement_columns_and_rows(self) -> tuple[set[int], set[int]]:
        """Returns the sets of columns and rows that contain a requirement"""
        cols, rows = tuple(map(set, zip(*self.requirement_cells())))
        return cols, rows

    def find_blank_columns_and_rows(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Returns a list of the indices of blank columns and
        a list of the indices of blank rows."""
        if self.dimensions == (0, 0):
            return tuple(), tuple()
        if not self.obstructions and not self.requirements:
            return tuple(range(self.dimensions[0])), tuple(range(self.dimensions[1]))
        not_blank_cols, not_blank_rows = zip(*self.not_blank_cells())
        blank_cols = tuple(set(range(self.dimensions[0])) - set(not_blank_cols))
        blank_rows = tuple(set(range(self.dimensions[1])) - set(not_blank_rows))
        return blank_cols, blank_rows

    def blank_and_near_blank(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Finds blank rows and cols allowing point row/col intersection"""
        if self.dimensions == (0, 0):
            return tuple(), tuple()
        if not self.obstructions and not self.requirements:
            return tuple(range(self.dimensions[0])), tuple(range(self.dimensions[1]))
        req_cells = tuple(
            chain(*(set(req.positions) for req in chain(*self.requirements)))
        )
        check_cells = (
            cell
            for cell in self.not_blank_cells()
            if cell in req_cells
            or (cell[0] not in self.point_cols and cell[1] not in self.point_rows)
        )
        not_blank_cols, not_blank_rows = zip(*check_cells)
        blank_cols = tuple(set(range(self.dimensions[0])) - set(not_blank_cols))
        blank_rows = tuple(set(range(self.dimensions[1])) - set(not_blank_rows))
        return blank_cols, blank_rows

    def remove_empty_rows_and_columns(self) -> "Tiling":
        """Deletes any rows and columns in the gridding that are empty"""
        empty_cols, empty_rows = self.find_empty_rows_and_columns()
        return self.delete_rows_and_columns(empty_cols, empty_rows)

    def remove_empty_columns(self) -> "Tiling":
        """Deletes any columns in the gridding that are empty"""
        empty_cols, _ = self.find_empty_rows_and_columns()
        return self.delete_columns(empty_cols)

    def sub_tiling(self, cells: Iterable[tuple[int, int]], simplify=True) -> "Tiling":
        """
        Returns a sub-tiling of the tiling at the given cells.
        """
        cells = set(cells)
        obstructions = []
        for ob in self.obstructions:
            if all(cell in cells for cell in ob.positions):
                obstructions.append(ob)
        requirements = []
        for req_list in self.requirements:
            new_req_list = []
            for req in req_list:
                if all(cell in cells for cell in req.positions):
                    new_req_list.append(req)
            if new_req_list:
                requirements.append(new_req_list)
                # assert len(new_req_list) == len(req_list)

        for cell in product(range(self.dimensions[0]), range(self.dimensions[1])):
            if cell not in cells:
                obstructions.append(GriddedCayleyPerm(CayleyPermutation([0]), [cell]))

        return Tiling(obstructions, requirements, self.dimensions, simplify=simplify)

    # Requirement insertion methods

    def remove_requirements(self, reqs: Iterable[GriddedCayleyPerm]) -> "Tiling":
        """
        Returns a new tiling with the given requirements removed. (requirements, not req lists)
        """
        new_requirements = []
        for req_list in self.requirements:
            new_req_list = [req for req in req_list if req not in reqs]
            if new_req_list:
                new_requirements.append(new_req_list)
        return Tiling(self.obstructions, new_requirements, self.dimensions)

    def add_obstructions(self, gcps: Iterable[GriddedCayleyPerm]) -> "Tiling":
        """
        Returns a new tiling with the given gridded Cayley permutations added as obstructions.
        """
        return Tiling(
            self.obstructions + tuple(gcps), self.requirements, self.dimensions
        )

    def add_obstruction(self, gcp: GriddedCayleyPerm) -> "Tiling":
        """
        Returns a new tiling with the given gridded Cayley permutation added as an obstruction.
        """
        return self.add_obstructions([gcp])

    def add_requirements(
        self, requirements: Iterable[Iterable[GriddedCayleyPerm]]
    ) -> "Tiling":
        """
        Returns a new tiling with the given requirements added.
        """
        return Tiling(
            self.obstructions, self.requirements + tuple(requirements), self.dimensions
        )

    def add_requirement_list(
        self, requirement_list: Iterable[GriddedCayleyPerm]
    ) -> "Tiling":
        """
        Returns a new tiling with the given requirement list added.
        """
        return self.add_requirements([requirement_list])

    @cached_property
    def point_rows(self) -> set[int]:
        """Returns the set of rows which only contain points of the same value."""
        return SimplifyObstructionsAndRequirements(
            self.obstructions, self.requirements, self.dimensions
        ).point_rows()

    @cached_property
    def point_cols(self) -> set[int]:
        """Returns the set of cols which can only contain one point."""
        point_cols = set[int]()
        obs = self.obs_by_col_and_row()[0]
        for col in range(self.dimensions[0]):
            if any(
                not (
                    {
                        GriddedCayleyPerm((0, 1), cells),
                        GriddedCayleyPerm((1, 0), reversed(cells)),
                    }.issubset(obs[col])
                )
                for cells in combinations_with_replacement(self.cells_in_col(col), 2)
            ):
                continue
            point_cols.add(col)
        return point_cols

    def cells_in_row(self, row: int) -> set[tuple[int, int]]:
        """Returns the set of active cells in the given row."""
        cells = set()
        for cell in self.active_cells:
            if cell[1] == row:
                cells.add(cell)
        return cells

    def cells_in_col(self, col: int) -> set[tuple[int, int]]:
        """Returns the set of active cells in the given column."""
        cells = set()
        for cell in self.active_cells:
            if cell[0] == col:
                cells.add(cell)
        return cells

    def col_is_positive(self, col: int) -> bool:
        """Return true if the column must contain at least one point."""
        req_list = tuple(
            GriddedCayleyPerm(CayleyPermutation([0]), [cell])
            for cell in self.cells_in_col(col)
        )
        return self.add_obstructions(req_list).is_empty()

    def row_is_positive(self, row: int) -> bool:
        """Return true if the row must contain at least one point."""
        req_list = tuple(
            GriddedCayleyPerm(CayleyPermutation([0]), [cell])
            for cell in self.cells_in_row(row)
        )
        return self.add_obstructions(req_list).is_empty()

    # Fusion methods

    def fuse(self, fuse_rows: bool, index: int) -> "Tiling":
        """If fuse_rows, tries to fuse rows, otherwise, tries to fuse cols."""
        if fuse_rows == 0:
            return self.delete_columns([index])
        return self.delete_rows([index])

    def can_fuse_row(self, index: int) -> bool:
        """Returns true if row at index is fusable"""
        return self.is_fusable(True, index)

    def can_fuse_col(self, index: int) -> bool:
        """Returns true if row at index is fusable"""
        return self.is_fusable(False, index)

    def is_fusable(self, fuse_rows: bool, index: int) -> bool:
        """If fuse rows, checks if the rows at index and index+1 are fuseable,
        otherwise does the same for cols at index and index+1."""
        if fuse_rows:
            test_tiling = self.delete_rows([index])
        else:
            test_tiling = self.delete_columns([index])
        test_tiling = test_tiling.split_row_or_col(fuse_rows, index)
        return test_tiling == self

    def split_row_or_col(self, unfuse_rows: bool, index: int) -> "Tiling":
        """Unfuses a row (if unfuse_rows) or col (otherwise)
        at index without simplifying the Tiling."""
        direction_map = {
            i: i - int(i > index) for i in range(self.dimensions[unfuse_rows] + 1)
        }
        identity_map = {i: i for i in range(self.dimensions[not unfuse_rows])}
        if unfuse_rows:
            new_map = RowColMap(identity_map, direction_map)
            new_dimensions = (self.dimensions[0], self.dimensions[1] + 1)
        else:
            new_map = RowColMap(direction_map, identity_map)
            new_dimensions = (self.dimensions[0] + 1, self.dimensions[1])
        new_obstructions = new_map.preimage_of_obstructions(self.obstructions)
        new_requirements = new_map.preimage_of_requirements(self.requirements)
        return Tiling(
            new_obstructions, new_requirements, new_dimensions, simplify=False
        )

    def split_point_row(self, row: int, above: bool) -> "Tiling":
        """Unfuses a point row at row 'row' either above or below that row."""
        unfused = self.split_row_or_col(True, row)
        point_row = row + int(above)
        return unfused.add_point_row(point_row)

    def add_point_row(self, point_row: int) -> "Tiling":
        """Adds obstructions to make the specified row a point row."""
        row_obs: list[GriddedCayleyPerm] = []
        for col in range(self.dimensions[0]):
            row_obs.append(
                GriddedCayleyPerm(
                    CayleyPermutation([0, 1]), [(col, point_row), (col, point_row)]
                )
            )
            row_obs.append(
                GriddedCayleyPerm(
                    CayleyPermutation([1, 0]), [(col, point_row), (col, point_row)]
                )
            )
        for col1, col2 in combinations(range(self.dimensions[0]), 2):
            row_obs.append(
                GriddedCayleyPerm(
                    CayleyPermutation([0, 1]), [(col1, point_row), (col2, point_row)]
                )
            )
            row_obs.append(
                GriddedCayleyPerm(
                    CayleyPermutation([1, 0]), [(col1, point_row), (col2, point_row)]
                )
            )
        return self.add_obstructions(row_obs)

    def is_point_row_fuseable(self, row: int) -> bool:
        """Returns true if row and row+1 are fuseable and either of them
        are point rows."""
        if row in self.point_rows:
            test_tiling = self.delete_rows([row]).split_point_row(row, False)
            if test_tiling == self:
                return True
        elif row + 1 in self.point_rows:
            test_tiling = self.delete_rows([row + 1]).split_point_row(row, True)
            if test_tiling == self:
                return True
        return False

    # Construction methods

    @classmethod
    def permutation_tiling(cls) -> "Tiling":
        """Returns the 1x1 tiling representing all permutations."""
        return cls(
            [GriddedCayleyPerm(CayleyPermutation((0, 0)), ((0, 0), (0, 0)))], [], (1, 1)
        )

    @classmethod
    def create_vincular_or_bivincular(
        cls,
        cperm: CayleyPermutation | str,
        adjacent_indices: Iterable[int] = [],
        adjacent_values: Iterable[int] = [],
    ) -> "Tiling":
        # pylint:disable=dangerous-default-value
        # pylint:disable=too-many-locals
        """
        Creates a tiling where the points in the Cayley permutation
        are point cells in the tiling. Input must be 0 based and cperm can
        be a Cayley permutation or a string.

        For creating a vincular pattern:
        Input an iterable of indices of the Cayley permutation which must be
        adjacent to adjacent_indices. Value i in adjacent_indices means positions
        i and i+1 must be adjacent. Indices must be 0 based.

        For creating a bivincular pattern:
        Input an iterable of values of the Cayley permutation which must be
        adjacent to adjacent_values. Value i in adjacent_values means values i and
        i+1 must be adjacent. Values must be 0 based.

        Note: Both adjacent_values and adjacent_indices can be used in conjunction
        or can both be left empty and for a Cayley permutation length n with maximum value m:
        - -1 in adjacent values implies there are no values below the smallest one
        - m in adjacent values implies there are no values above the largest one
        - -1 in adjacent indices implies there are no values to the left of the leftmost one
        - n in adjacent indices implies there are no values to the right of the rightmost one

        Example:
        >>> cperm = "012"
        >>> til = MappedTiling.create_vincular_or_bivincular(cperm, adjacent_indices=[0])
        >>> print(til)
        +-+-+-+-+-+-+
        | |#|#| |#| |
        +-+-+-+-+-+-+
        |0|#|#|0|●|0|*
        +-+-+-+-+-+-+
        | |#|#| |#| |
        +-+-+-+-+-+-+
        |0|#|●|0|#|0|*
        +-+-+-+-+-+-+
        | |#|#| |#| |
        +-+-+-+-+-+-+
        |0|●|#|0|#|0|*
        +-+-+-+-+-+-+
        | |#|#| |#| |
        +-+-+-+-+-+-+
        Key:
        0: Av(01,10)
        Crossing obstructions:
        Requirements 0:
        0: (1, 1)
        Requirements 1:
        0: (2, 3)
        Requirements 2:
        0: (4, 5)
        """
        if isinstance(cperm, str):
            cperm = string_to_basis(cperm)[0]
        dimensions = (len(cperm), max(cperm) + 1)
        perm_cells = [(2 * k + 1, 2 * cperm[k] + 1) for k in range(dimensions[0])]
        all_obs, all_reqs = [], []
        cols, rows = [2 * i + 1 for i in range(dimensions[0])], [
            2 * i + 1 for i in range(dimensions[1])
        ]
        col_cells = [
            cell
            for cell in product(cols, range(2 * dimensions[1] + 1))
            if cell not in perm_cells
        ]  # this is all the cells that will have point obstructions
        for cell in col_cells:
            all_obs.append(GriddedCayleyPerm(CayleyPermutation([0]), [cell]))
        for i in rows:  # this puts the 01,10 obstructions across each row
            for j in range(2 * dimensions[0] + 1):
                cell1 = (j, i)
                if cell1 not in col_cells:
                    for k in range(j, 2 * dimensions[0] + 1):
                        cell2 = (k, i)
                        if cell2 not in col_cells:
                            all_obs.append(
                                GriddedCayleyPerm(
                                    CayleyPermutation([0, 1]), [cell1, cell2]
                                )
                            )
                            all_obs.append(
                                GriddedCayleyPerm(
                                    CayleyPermutation([1, 0]), [cell1, cell2]
                                )
                            )
        for (
            cell
        ) in (
            perm_cells
        ):  # this puts the point requirement and the 00 obstruction according to the permutation
            all_reqs.append([GriddedCayleyPerm(CayleyPermutation([0]), [cell])])
            all_obs.append(GriddedCayleyPerm(CayleyPermutation([0, 0]), [cell, cell]))
        return cls(
            all_obs, all_reqs, (2 * dimensions[0] + 1, 2 * dimensions[1] + 1)
        ).delete_rows_and_columns(
            cols=[i * 2 + 2 for i in adjacent_indices],
            rows=[i * 2 + 2 for i in adjacent_values],
        )

    def create_mesh_pattern(
        self,
        shaded_regions: Iterable[tuple[Cell, Cell] | tuple[Cell]],
    ) -> "Tiling":
        """Creates a mesh pattern from a tiling and an input of cells or
        tuples of two cells such that the area between each pair of cells must be empty.
        A cell is a tuple of two integers, everything must be 0 based.

        The function Tiling.from_vincular_or_bivincular(cperm) can be used to create
        a tiling from cperm, a Cayley permutation from which the 0 based cells
        can be read off.

        Example:
        >>> cperm = CayleyPermutation([0, 1, 2])
        >>> til = MappedTiling.create_vincular_or_bivincular()
        >>> print(til.create_mesh_pattern([[(0, 2), (2, 4)]]))
        +-+-+-+-+-+-+-+
        | |#| |#| |#| |
        +-+-+-+-+-+-+-+
        |0|#|0|#|0|●|0|*
        +-+-+-+-+-+-+-+
        |#|#|#|#| |#| |
        +-+-+-+-+-+-+-+
        |#|#|#|●|0|#|0|*
        +-+-+-+-+-+-+-+
        |#|#|#|#| |#| |
        +-+-+-+-+-+-+-+
        |0|●|0|#|0|#|0|*
        +-+-+-+-+-+-+-+
        | |#| |#| |#| |
        +-+-+-+-+-+-+-+
        Key:
        0: Av(01,10)
        Crossing obstructions:
        Requirements 0:
        0: (1, 1)
        Requirements 1:
        0: (3, 3)
        Requirements 2:
        0: (5, 5)
        """
        dim_n, dim_m = self.dimensions
        gcps = []
        for cells in shaded_regions:
            if len(cells) == 1:
                cell = cells[0]
                if cell[0] > dim_n or cell[1] > dim_m:
                    raise ValueError(
                        "Cells must be at most the dimensions of the tiling"
                    )
                gcps.append(GriddedCayleyPerm(CayleyPermutation([0]), cells))
            elif len(cells) == 2:
                max_n = max(cells[0][0], cells[1][0])
                max_m = max(cells[0][1], cells[1][1])
                if max_n > dim_n or max_m > dim_m:
                    raise ValueError(
                        "Cells must be at most the dimensions of the tiling"
                    )
                for n in range(min(cells[0][0], cells[1][0]), max_n + 1):
                    for m in range(min(cells[0][1], cells[1][1]), max_m + 1):
                        gcps.append(GriddedCayleyPerm(CayleyPermutation([0]), [(n, m)]))
            else:
                raise ValueError(
                    "Input to shaded_regions must be either cells or tuples of two cells such that "
                    + "the shaded region is between the two cells."
                )
        return self.add_obstructions(gcps)

    @staticmethod
    def from_vincular_with_obs(
        cperm: CayleyPermutation, adjacencies: Iterable[int]
    ) -> "Tiling":
        """
        Both cperm and adjacencies must be 0 based. Creates a tiling from a
        vincular pattern such that the adjacent columns have obstructions.
        Adjacencies is a list of positions where i in
        adjacencies means positions i and i+1 must be adjacent.
        """
        dimensions = (len(cperm), max(cperm) + 1)
        all_obs, all_reqs = [], []
        perm_cells = [(2 * k + 1, 2 * cperm[k] + 1) for k in range(dimensions[0])]
        cols, rows = [2 * i + 1 for i in range(dimensions[0])] + [
            2 * i + 2 for i in adjacencies
        ], [2 * i + 1 for i in range(dimensions[1])]
        col_cells = [
            cell
            for cell in product(cols, range(2 * dimensions[1] + 1))
            if cell not in perm_cells
        ]  # this is all the cells that will have point obstructions
        for cell in col_cells:
            all_obs.append(GriddedCayleyPerm(CayleyPermutation([0]), [cell]))
        for i in rows:  # this puts the 01,10 obstructions across each row
            for j in range(2 * dimensions[0] + 1):
                cell1 = (j, i)
                if cell1 not in col_cells:
                    for k in range(j, 2 * dimensions[0] + 1):
                        cell2 = (k, i)
                        if cell2 not in col_cells:
                            all_obs.append(
                                GriddedCayleyPerm(
                                    CayleyPermutation([0, 1]), [cell1, cell2]
                                )
                            )
                            all_obs.append(
                                GriddedCayleyPerm(
                                    CayleyPermutation([1, 0]), [cell1, cell2]
                                )
                            )
        for (
            cell
        ) in (
            perm_cells
        ):  # this puts the point requirement and the 00 obstruction according to the permutation
            all_reqs.append([GriddedCayleyPerm(CayleyPermutation([0]), [cell])])
            all_obs.append(GriddedCayleyPerm(CayleyPermutation([0, 0]), [cell, cell]))
        return Tiling(all_obs, all_reqs, (2 * dimensions[0] + 1, 2 * dimensions[1] + 1))

    def compare_to(
        self, other: "Tiling", depth: int = 4
    ) -> tuple[
        bool, Optional[tuple[list[int], list[int]]], Optional[GriddedCayleyPerm]
    ]:
        """Compares the gcps that live on self to the gcps on other up to size depth"""
        original = self.initial_conditions(depth)
        new = other.initial_conditions(depth)
        return (original == new, (original, new), None)

    # CSS methods

    def to_jsonable(self) -> dict:
        res = {
            "obstructions": [ob.to_jsonable() for ob in self.obstructions],
            "requirements": [
                [req.to_jsonable() for req in req_list]
                for req_list in self.requirements
            ],
            "dimensions": self.dimensions,
        }
        res.update(super().to_jsonable())
        return res

    @classmethod
    def from_dict(cls, d: dict) -> "Tiling":
        return Tiling(
            [GriddedCayleyPerm.from_dict(ob) for ob in d["obstructions"]],
            [
                [GriddedCayleyPerm.from_dict(req) for req in req_list]
                for req_list in d["requirements"]
            ],
            d["dimensions"],
        )

    def maximum_length_of_minimal_gridded_cayley_perm(self) -> int:
        """Return an upper bound on the length of a minimal gridded Cayley permutation."""
        return sum(max(len(gcp) for gcp in req_list) for req_list in self.requirements)

    def is_empty(self) -> bool:
        return any(len(ob) == 0 for ob in self.obstructions) or self._is_empty()

    def _is_empty(self) -> bool:
        for _ in self.minimal_gridded_cperms():
            return False
        return True

    def is_horizontal_insertion_encodable(self) -> bool:
        """Returns True if the tiling has a horizontal insertion encoding."""
        if self.dimensions[0] == 1:
            patterns_in_cells: list[tuple[CayleyPermutation, ...]] = []
            for cell in self.active_cells:
                patterns_in_cells.append(
                    tuple(
                        gcp.pattern
                        for gcp in self.obstructions
                        if all(c[1] == cell[1] for c in gcp.positions)
                    )
                )
            if all(
                regular_horizontal_insertion_encoding(patterns_in_cell)
                for patterns_in_cell in patterns_in_cells
            ):
                return True
        return False

    def is_vertical_insertion_encodable(self) -> bool:
        """Returns True if the tiling has a vertical insertion encoding."""
        if self.dimensions[1] == 1:
            patterns_in_cells: list[tuple[CayleyPermutation, ...]] = []
            for cell in self.active_cells:
                patterns_in_cells.append(
                    tuple(
                        gcp.pattern
                        for gcp in self.obstructions
                        if all(c[0] == cell[0] for c in gcp.positions)
                    )
                )
            if all(
                regular_vertical_insertion_encoding(patterns_in_cell)
                for patterns_in_cell in patterns_in_cells
            ):
                return True
        return False

    def is_subset(self, other: "Tiling") -> bool:
        """
        Return True if the set of gridded permutations on self
        is a subset of the set of gridded permutations on other.
        """
        if not self.dimensions == other.dimensions:
            return False
        return set(other.obstructions).issubset(set(self.obstructions)) and set(
            self.requirements
        ).issubset(set(other.requirements))

    def minimal_gridded_cperms(self) -> Iterator[GriddedCayleyPerm]:
        """Returns an iterator of minimal gridded Cayley permutations."""
        if self.requirements:
            yield from MinimalGriddedCayleyPerm(
                self.obstructions, self.requirements
            ).minimal_gridded_cperms()
        else:
            yield GriddedCayleyPerm(CayleyPermutation([]), [])

    def is_atom(self) -> bool:
        """Return True if tiling is a single gridded permutation."""
        return (
            (self.active_cells == self.point_cells())
            and self.fully_isolated()
            and not any(len(ob.positions) == 0 for ob in self.obstructions)
        )

    def fully_isolated(self) -> bool:
        """Check if all cells are isolated on their rows and columns."""
        seen_col: list[int] = []
        point_rows = self.point_rows
        for i, j in self.active_cells:
            if i in seen_col or j not in point_rows:
                return False
            seen_col.append(i)
        return True

    def minimum_size_of_object(self) -> int:
        """Or a lower bound."""
        if self.is_empty():
            return 0
        i = 0
        while True:
            for _ in self.objects_of_size(i):
                return i
            i += 1

    def objects_of_size(self, n: int, **parameters: int) -> Iterator[GriddedCayleyPerm]:
        yield from self.gridded_cayley_permutations(n)

    @cached_property
    def cell_basis(
        self,
    ) -> dict[Cell, tuple[list[CayleyPermutation], list[CayleyPermutation]]]:
        """Returns a dictionary from cells to basis.

        The basis for each cell is a tuple of two lists of permutations.  The
        first list contains the patterns of the obstructions localized in the
        cell and the second contains the intersections of requirement lists
        that are localized in the cell.
        """
        obdict: dict[Cell, list[CayleyPermutation]] = defaultdict(list)
        reqdict: dict[Cell, list[CayleyPermutation]] = defaultdict(list)
        for ob in self.obstructions:
            if len(set(ob.positions)) == 1:
                cell = ob.positions[0]
                obdict[cell].append(ob.pattern)

        for req_list in self.requirements:
            for gcp in req_list:
                for cell in set(gcp.positions):
                    subgcp = gcp.get_gridded_perm_in_cells([cell])
                    if subgcp not in reqdict[cell] and all(
                        r.contains_gridded_cperm(subgcp) for r in req_list
                    ):
                        reqdict[cell].append(subgcp.pattern)
        for cell, contain in reqdict.items():
            ind_to_remove = set()
            for i, req in enumerate(contain):
                if any(req in other for j, other in enumerate(contain) if i != j):
                    ind_to_remove.add(i)
            reqdict[cell] = [
                req for i, req in enumerate(contain) if i not in ind_to_remove
            ]

        all_cells = product(range(self.dimensions[0]), range(self.dimensions[1]))
        resdict = {cell: (obdict[cell], reqdict[cell]) for cell in all_cells}
        return resdict

    @cached_property
    def cell_labels(self) -> dict[Cell, str]:
        """Assigns each cell a labelaccording to cell basis"""
        used_labels = dict[tuple[tuple[CayleyPermutation, ...], bool], str]()
        labels = dict[Cell, str]()
        curr_label = 1
        for cell, gridded_perms in sorted(self.cell_basis.items()):
            obstructions, _ = gridded_perms
            basis = list(sorted(obstructions))

            # the block, is the basis and whether or not positive
            block = (tuple(basis), cell in self.positive_cells())
            label = used_labels.get(block)
            if label is None:

                match basis:
                    case [CayleyPermutation((0,))] | []:
                        label = ""
                        continue
                    case [
                        CayleyPermutation((0, 0)),
                        CayleyPermutation((0, 1)),
                        CayleyPermutation((1, 0)),
                    ]:
                        if cell in self.positive_cells():
                            label = "\u25cf"
                        else:
                            label = "\u25cb"
                        # if cell[1] in self.point_rows:
                        #     label = "-" + label + "-"
                    case [
                        CayleyPermutation((0, 1)),
                        CayleyPermutation((1, 0)),
                    ]:
                        label = "-"
                    case [CayleyPermutation((0, 1))]:
                        label = "\\"
                    case [CayleyPermutation((1, 0))]:
                        label = "/"
                    case [CayleyPermutation((0, 0))]:
                        label = "="
                    case _:
                        label = chr(ord("@") + curr_label)
                        curr_label += 1
                used_labels[block] = label
            labels[cell] = label
        return labels

    # html methods

    def _html_table(self) -> list[str]:
        """Returns an the list of strings used to make the html representation.
        There is an invisible row and col to make Mappling html easier."""
        # pylint: disable=too-many-locals
        # stylesheet for tiling
        empty_cells = self.empty_cells()
        style = """
            border: 1px solid;
            width: 24px;
            height: 24px;
            text-align: center;
            """
        empty = "background-color : grey;"
        dim_i, dim_j = self.dimensions
        result = []
        # Create tiling html table, has one extra row/col for RowColMap
        result.append('<table style = "margin-left:auto; margin-right:auto">')
        for i in range(dim_j):
            result.append("<tr>")
            for j in range(dim_i):
                temp_style = style + empty * ((j, dim_j - i - 1) in empty_cells)
                result.append(f"<th style='{temp_style}'>")
                result.append(" ")
                result.append("</th>")
            result.append("</tr>")
        result.append("</table>")

        cell_labels = self.cell_labels
        # How many characters are in a row in the grid
        row_width = 3 * dim_i + 2
        for cell, label in cell_labels.items():

            row_index_from_top = dim_j - cell[1] - 1
            index = row_index_from_top * row_width + cell[0] * 3 + 3
            result[index] = label
        return result

    def to_html_representation(self):
        return "".join(self._html_table())

    @classmethod
    def empty_tiling(cls) -> "Tiling":
        """Return the tiling that is the empty set."""
        return cls([GriddedCayleyPerm(CayleyPermutation([]), [])], [], (0, 0))

    def copy(self) -> "Tiling":
        """Return a copy of the tiling."""
        return Tiling(self.obstructions, self.requirements, self.dimensions)

    def __repr__(self) -> str:
        return (
            f"Tiling({repr(self.obstructions)}, {repr(self.requirements)},"
            + f"{repr(self.dimensions)})"
        )

    def _string_table(self) -> list[str]:
        """Creates a list of strings for each row of the __str__ grid"""
        if self.dimensions == (0, 0):
            return ["┌ ┐", " ε ", "└ ┘"]
        cell_labels = self.cell_labels
        for cell in self.empty_cells():
            cell_labels[cell] = "░"
            if cell in self.point_rows:
                cell_labels[cell] = "#"
        row_separator = "├" + ("┼─" * self.dimensions[0] + "┤")[1:]
        top_row = "┌" + ("┬─" * self.dimensions[0])[1:] + "┐"
        bottom_row = "└" + ("┴─" * self.dimensions[0])[1:] + "┘"
        final_table = [row_separator]
        for row in range(self.dimensions[1]):
            new_row = "│"
            for col in range(self.dimensions[0]):
                label = " "
                if (col, row) in cell_labels:
                    label = cell_labels[(col, row)]
                new_row += label + "│"
            if row in self.point_rows:
                new_row += "*"
            final_table += [new_row, row_separator]
        final_table.reverse()
        final_table[0] = top_row
        final_table[-1] = bottom_row
        return final_table

    def __str__(self) -> str:
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals

        all_cayley_obs = set[GriddedCayleyPerm]()
        point_cells = self.point_cells()
        if self.dimensions != (0, 0):
            is_perm_tiling = True
            for row, cols in product(
                range(self.dimensions[1]),
                combinations_with_replacement(range(self.dimensions[0]), 2),
            ):
                if cols[0] == cols[1] and (cols[0], row) in point_cells:
                    continue
                if self.active_cells.issuperset({(cols[0], row), (cols[1], row)}):
                    cayley_ob = GriddedCayleyPerm(
                        (0, 0), ((cols[0], row), (cols[1], row))
                    )
                    if cayley_ob not in self.obstructions:
                        is_perm_tiling = False
                        break
                    all_cayley_obs.add(cayley_ob)
            if is_perm_tiling and all_cayley_obs:
                new_obs = set(self.obstructions) - all_cayley_obs
                return "Permutation Tiling\n" + str(
                    Tiling(new_obs, self.requirements, self.dimensions)
                )
        final_string = "\n".join(self._string_table())

        key_dict = dict[str, list[CayleyPermutation]]()
        for cell, label in self.cell_labels.items():
            if not label.isalpha():
                continue
            if label not in key_dict:
                key_dict[label] = self.cell_basis[cell][0]
        if key_dict:
            key_string = "\nKey: \n"
            for label, patts in key_dict.items():
                basis_string = ", ".join(map(str, patts))
                key_string += f"{label}: Av({basis_string}) \n"
            final_string += key_string

        if self == Tiling.empty_tiling():
            final_string += "Obstructions: ε"

        if self.requirements:
            requirements_string = "\n"
            for i, req_list in enumerate(self.requirements):
                requirements_string += f"Requirements {i}: \n"
                for req in req_list:
                    requirements_string += f"{req} \n"
            final_string += requirements_string

        pr_implied = (CayleyPermutation((0, 1)), CayleyPermutation((1, 0)))
        crossing_obs = set[GriddedCayleyPerm]()
        for ob in self.obstructions:
            if len(set(ob.positions)) == 1:
                continue
            if (
                ob.pattern in pr_implied
                and ob.positions[0][1] == ob.positions[1][1]
                and ob.positions[0][1] in self.point_rows
            ):
                continue
            crossing_obs.add(ob)

        if len(crossing_obs) > 0:
            crossing_string = "\nCrossing obstructions: \n"
            crossing_string += "\n".join(map(str, sorted(crossing_obs)))
            final_string += crossing_string

        return final_string

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tiling):
            return NotImplemented
        return (
            self.obstructions == other.obstructions
            and self.requirements == other.requirements
            and self.dimensions == other.dimensions
        )

    def __hash__(self) -> int:
        return hash((self.obstructions, self.requirements, self.dimensions))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Tiling):
            return NotImplemented
        return self.dimensions < other.dimensions or (
            self.dimensions == other.dimensions
            and len(self.obstructions) < len(other.obstructions)
        )

    def __leq__(self, other: object) -> bool:
        if not isinstance(other, Tiling):
            return NotImplemented
        return self < other or self == other
