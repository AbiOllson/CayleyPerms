"""For row and column separating a tiling and related strategies."""

import heapq
import abc
from collections import defaultdict
from itertools import combinations, product
from typing import Iterator, Optional, Generic, Tuple, TypeVar, Iterable
from functools import cached_property
from comb_spec_searcher import DisjointUnionStrategy, StrategyFactory
from comb_spec_searcher.exception import StrategyDoesNotApply

from gridded_cayley_permutations.row_col_map import RowColMap
from gridded_cayley_permutations import Tiling, GriddedCayleyPerm
from gridded_cayley_permutations.point_placements import TilingT
from cayley_permutations import CayleyPermutation

Cell = Tuple[int, int]
Obstructions = Tuple[GriddedCayleyPerm, ...]
Requirements = Tuple[Obstructions, ...]


Vertex = TypeVar("Vertex", bound=object)


class Graph(Generic[Vertex]):
    """
    A weighted directed graph implemented with an adjacency matrix.

    The graph is made such that it is easy to merge to vertices. Merging
    vertices collapse to vertices together such that
        - The weight of the new vertex is the sum of the weights
        - The weight of the edges is the sum of the weight of the edges that
        went to any of the merged vertices before the merge.


    The graph supports 2 operations
        - `reduce`: who merge two vertices that were not connected by an edges
        and repeat as long as possible.
        - `break_cycle_in_all_ways`: Take a cycle in the graph and return a
        copy of the graph with a removed edges for each edges in the cycle.

    Moreover, one can also ask:
        - if the graph is acyclic with `is_acyclic`
        - for a cycle of the graph with `find_cycle`
        - For the vertex order implied by a reduced acyclic graph
    """

    def __init__(self, vertices: Iterable[Vertex], matrix=None):
        self._vertex_labels = [set([v]) for v in vertices]
        self._vertex_weights = [1 for _ in self._vertex_labels]
        self._matrix = matrix
        assert len(matrix) == len(self._vertex_labels)
        assert all(len(row) == len(self._matrix) for row in matrix)
        self._reduced = False
        self._is_acyclic = False

    @property
    def num_vertices(self) -> int:
        """
        The number of vertices of the graph
        """
        return len(self._vertex_weights)

    def _merge_vertices(self, v1: int, v2: int) -> None:
        """
        Merge the two vertices.

        Vertex and edges are merged and the weight are added. Then edges with a
        weight that is to small are discarded.
        """
        v2_label = self._vertex_labels.pop(v2)
        self._vertex_labels[v1].update(v2_label)
        v2_weight = self._vertex_weights.pop(v2)
        self._vertex_weights[v1] += v2_weight
        self._add_matrix_rows(v1, v2)
        self._add_matrix_columns(v1, v2)
        self._trim_edges(v1)

    def reduce(self) -> None:
        """Reduce the graph."""
        if self._reduced:
            return
        non_edge = self.find_non_edge()
        while non_edge:
            self._merge_vertices(non_edge[0], non_edge[1])
            non_edge = self.find_non_edge()
        self._reduced = True

    def find_non_edge(self) -> Optional[tuple[int, int]]:
        """
        Return a non-edge of the graph.

        A non edges is a pair of vertices `(v1, v2)` such that neither
        `(v1, v2)` or `(v2, v1)` is an edge in the graph.
        """
        for v1, v2 in combinations(range(self.num_vertices), 2):
            if not self._is_edge(v1, v2) and not self._is_edge(v2, v1):
                return (v1, v2)
        return None

    def is_acyclic(self) -> bool:
        """
        Check if the graph is acyclic.

        To perform that check, the graph must first be reduced with the
        `reduce` method.
        """
        assert self._reduced, "Graph must first be reduced"
        if self._is_acyclic or self.num_vertices == 0:
            return True
        return self.find_cycle() is None

    def find_cycle(self) -> Optional[tuple[tuple[int, int], ...]]:
        """
        Return the edges of a cycle of the graphs. The graphs first need to be
        reduced

        If a cycle of length 3 is return it means that no cycle of length 2
        exist.

        If the graph is acyclic, returns None.

        NOTE:

            One can prove that if a reduced graph is not acyclic it has either
            a cycle of length 2 or 3.
        """
        assert self._reduced, "Graph must first be reduced"
        for v1, v2 in combinations(range(self.num_vertices), 2):
            if self._is_edge(v1, v2) and self._is_edge(v2, v1):
                return ((v1, v2), (v2, v1))
        for v1, v2, v3 in combinations(range(self.num_vertices), 3):
            cycle = self._length3_cycle(v1, v2, v3)
            if cycle:
                return cycle
        self._is_acyclic = True
        return None

    def break_cycle_in_all_ways(
        self, edges: Iterable[tuple[int, int]]
    ) -> Iterator["Graph[Vertex]"]:
        """
        Generator over Graph object obtained by removing one edge of the
        `edges` iterator.
        """
        # pylint: disable=protected-access
        for e in edges:
            new_graph = Graph.__new__(Graph)
            new_graph._vertex_labels = [vl.copy() for vl in self._vertex_labels]
            new_graph._vertex_weights = self._vertex_weights.copy()
            new_graph._matrix = [row.copy() for row in self._matrix]
            new_graph._matrix[e[0]][e[1]] = 0
            new_graph._reduced = False
            new_graph._is_acyclic = False
            yield new_graph

    def vertex_order(self) -> list[set[Vertex]]:
        """
        Return the order of the vertex in a reduced acyclic graph.

        A reduced acyclic graph is an acyclic orientation of a complete graph.
        There it equivalent to an ordering of its vertices.

        To compute the vertex order, the graph must be reduced and acyclic.
        """
        assert self._reduced, "Graph must first be reduced"
        assert self.is_acyclic(), "Graph must be acyclic"
        vert_num_parent = [row.count(0) for row in self._matrix]
        return [p[1] for p in sorted(zip(vert_num_parent, self._vertex_labels))]

    def _add_matrix_rows(self, row1_idx: int, row2_idx: int) -> None:
        """
        Deletes row 2 from the graph matrix and change row 1 to
        the sum of both row.
        """
        assert row1_idx != row2_idx
        row1 = self._matrix[row1_idx]
        row2 = self._matrix.pop(row2_idx)
        self._matrix[row1_idx] = list(map(sum, zip(row1, row2)))

    def _add_matrix_columns(self, col1_idx: int, col2_idx: int) -> None:
        """
        Deletes column 2 from the graph matrix and change column 1 to
        the sum of both column.
        """
        assert col1_idx != col2_idx
        for row in self._matrix:
            c2_value = row.pop(col2_idx)
            row[col1_idx] += c2_value

    def _trim_edges(self, vertex: int) -> None:
        """
        Remove all the edges that touch vertex that that have a weight which is
        too small.

        The weight of an edge is too small if it is smaller than the product
        of the weights of the two vertex it connects.
        """
        v1 = vertex
        v1_weight = self._vertex_weights[v1]
        for v2 in range(self.num_vertices):
            v2_weight = self._vertex_weights[v2]
            weight_prod = v1_weight * v2_weight
            self._delete_edge_if_small(v1, v2, weight_prod)
            self._delete_edge_if_small(v2, v1, weight_prod)

    def _delete_edge_if_small(self, head: int, tail: int, cap: int) -> None:
        """
        Delete the edges that goes from head to tail if its weight is lower
        than the cap.
        """
        weight = self._matrix[head][tail]
        if weight < cap:
            self._matrix[head][tail] = 0

    def _is_edge(self, vertex1: int, vertex2: int) -> bool:
        """Check if the edge (vertex1, vertex2) is in the graph."""
        return self._matrix[vertex1][vertex2] != 0

    def _length3_cycle(
        self, v1: int, v2: int, v3: int
    ) -> Optional[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
        """
        Return the edges of a length 3 cycle containing the three vertices if
        such a cycle exist. Otherwise return None
        """

        def is_cycle(edges):
            return all(self._is_edge(*e) for e in edges)

        orientation1 = ((v1, v2), (v2, v3), (v3, v1))
        if is_cycle(orientation1):
            return orientation1
        orientation2 = ((v1, v3), (v3, v2), (v2, v1))
        if is_cycle(orientation2):
            return orientation2
        return None

    def __repr__(self) -> str:
        """Return a string representation of the graph."""
        s = f"Graph over the vertices {self._vertex_labels}\n"
        s += f"Vertex weight is {self._vertex_weights}\n"
        for row in self._matrix:
            s += f"{row}\n"
        return s

    def __lt__(self, other: "Graph") -> bool:
        """
        A graph is 'smaller if it as more vertices.
        Useful for the priority queue
        """
        return self.num_vertices > other.num_vertices

    def __le__(self, other: "Graph") -> bool:
        """
        A graph is 'smaller if it as more vertices.
        Useful for the priority queue
        """
        return self.num_vertices >= other.num_vertices


class RowColOrder:
    """Class to compute the order of the cells in a tiling."""

    def __init__(
        self,
        cells: set[Cell],
        col_ineq: set[tuple[Cell, Cell]],
        row_ineq: set[tuple[Cell, Cell]],
    ):
        self._active_cells = tuple(sorted(cells))
        self.row_ineq = row_ineq
        self.col_ineq = col_ineq

    def cell_at_idx(self, idx: int) -> Cell:
        """Return the cell at index `idx`."""
        return self._active_cells[idx]

    def cell_idx(self, cell: Cell) -> int:
        """Return the index of the cell"""
        return self._active_cells.index(cell)

    def _basic_matrix(self, row: bool) -> list[list[int]]:
        """
        Compute the basic matrix of inequalities based only on difference in
        row and columns. If `row` is True return the matrix for the row,
        otherwise return if for the columns.
        """
        idx = 1 if row else 0
        m = []
        for c1 in self._active_cells:
            new_row = [1 if c1[idx] < c2[idx] else 0 for c2 in self._active_cells]
            m.append(new_row)
        return m

    def _add_ineq(self, ineq: tuple[Cell, Cell], matrix: list[list[int]]) -> None:
        """
        Add an inequalities to the matrix.

        The inequalities must a tuple (smaller_cell, bigger_cell).
        """
        small_c, big_c = ineq
        matrix[self.cell_idx(small_c)][self.cell_idx(big_c)] = 1

    @cached_property
    def _ineq_matrices(self) -> tuple[list[list[int]], list[list[int]]]:
        """
        Return the matrices of inequalities between the cells.

        OUTPUT:
            tuple `(row_matrix, col_matrix)`
        """
        row_m = self._basic_matrix(row=True)
        col_m = self._basic_matrix(row=False)
        for ineq in self.row_ineq:
            self._add_ineq(ineq, row_m)
        for ineq in self.col_ineq:
            self._add_ineq(ineq, col_m)
        return row_m, col_m

    def row_ineq_graph(self) -> Graph[Cell]:
        """Return the graph of the row inequalities."""
        return Graph(self._active_cells, self._ineq_matrices[0])

    def col_ineq_graph(self) -> Graph[Cell]:
        """Return the graph of the column inequalities."""
        return Graph(self._active_cells, self._ineq_matrices[1])

    @staticmethod
    def _all_order(
        graph: Graph[Cell], only_max: bool = False
    ) -> Iterator[list[set[Cell]]]:
        """
        Generator of ordering of the active cells.

        One can get only the maximal separation by setting `only_max` to
        `True`.

        The order are yielded in decreasing order of size.
        """
        max_sep_seen = 0
        graph.reduce()
        heap = [graph]
        while heap and (not only_max or max_sep_seen <= graph.num_vertices):
            graph = heapq.heappop(heap)
            cycle = graph.find_cycle()
            if cycle is None:
                yield graph.vertex_order()
            else:
                for g in graph.break_cycle_in_all_ways(cycle):
                    g.reduce()
                    heapq.heappush(heap, g)

    @staticmethod
    def _maximal_order(graph: Graph[Cell]) -> list[set[Cell]]:
        """Returns a order that maximises separation."""
        return next(RowColOrder._all_order(graph))

    @cached_property
    def max_row_order(self) -> list[set[Cell]]:
        """A maximal order on the rows."""
        return self._maximal_order(self.row_ineq_graph())

    @cached_property
    def max_col_order(self) -> list[set[Cell]]:
        """A maximal order on the columns."""
        return self._maximal_order(self.col_ineq_graph())

    @cached_property
    def max_column_row_order(
        self,
    ) -> tuple[list[set[Cell]], list[set[Cell]]]:
        """A maximal order on the rows and columns."""
        return self.max_col_order, self.max_row_order


class AbstractSeparation:
    """Abstract class for row and column separation."""

    def __init__(
        self,
        tiling: Tiling,
        row_order: Optional[list[set[Cell]]] = None,
    ) -> None:
        """Initialize the separation algorithm with a tiling."""
        self.tiling = tiling
        if not row_order:
            row_order = self.max_row_col_order[1]
        self.row_order = row_order
        self.col_order = self.max_row_col_order[0]

    @property
    def max_row_col_order(self) -> tuple[list[set[Cell]], list[set[Cell]]]:
        """Return the column and row order of the tiling (columns first)."""
        col_ineq, row_ineq = self.column_row_inequalities()
        col_order, row_order = RowColOrder(
            self.tiling.active_cells, col_ineq, row_ineq
        ).max_column_row_order
        return col_order, row_order

    @abc.abstractmethod
    def point_row_obs_and_reqs(
        self,
    ) -> Iterator[tuple[Obstructions, Requirements]]:
        """
        Return the obstructions and requirements for the points in the rows.
        """

    @property
    def new_obstructions(self) -> Obstructions:
        """The new obstructions for the tiling"""
        new_obstructions = []
        for cell in product(
            range(self.new_dimensions[0]), range(self.new_dimensions[1])
        ):
            if cell not in self.new_active_cells:
                new_obstructions.append(
                    GriddedCayleyPerm(CayleyPermutation([0]), (cell,))
                )
        return tuple(new_obstructions)

    @property
    @abc.abstractmethod
    def new_active_cells(self) -> set[Cell]:
        """Return the new active cells of the tiling."""

    @property
    def new_dimensions(self) -> tuple[int, int]:
        """Return the new dimensions of the tiling."""
        return (len(self.row_col_map.col_map), len(self.row_col_map.row_map))

    @property
    @abc.abstractmethod
    def row_col_map(self) -> RowColMap:
        """Return the row and column map."""

    @abc.abstractmethod
    def map_cell(self, cell: Cell) -> Cell:
        """
        Map the cell to its new position.
        """

    @abc.abstractmethod
    def row_col_separation(self) -> Iterator[Tiling]:
        """
        Return the tiling with the row and column separated.
        """

    def inequalities_sets(
        self,
    ) -> tuple[set[tuple[Cell, Cell]], set[tuple[Cell, Cell]], set[tuple[Cell, Cell]]]:
        """Finds the length 2 obstructions in different cells.
        If they are on the same column and are an increasing obstruction,
        they are added to less_than_col to separate columns.
        If they are on the same row and are an increasing obstruction,
        they are added to less_than_row to separate rows.
        If they are in the same row and a constant obstruction,
        they are added to not_equal to help with strictly less than later.
        """
        not_equal = set()
        less_than_row = set()
        less_than_col = set()
        for ob in self.tiling.obstructions:
            if len(ob) == 2:
                cell1, cell2 = ob.positions
                if cell1 == cell2:
                    continue
                if cell1[0] == cell2[0]:
                    if ob.pattern == CayleyPermutation([0, 1]):
                        less_than_col.add((cell2, cell1))
                    if ob.pattern == CayleyPermutation([1, 0]):
                        less_than_col.add((cell2, cell1))
                elif cell1[1] == cell2[1]:
                    if ob.pattern == CayleyPermutation([0, 1]):
                        less_than_row.add((cell2, cell1))
                    if ob.pattern == CayleyPermutation([1, 0]):
                        if (cell2, cell1) in less_than_row:
                            less_than_row.remove((cell2, cell1))
                        else:
                            less_than_row.add((cell1, cell2))
                    if ob.pattern == CayleyPermutation([0, 0]):
                        not_equal.add((cell1, cell2))
                        not_equal.add((cell2, cell1))
        return less_than_col, less_than_row, not_equal

    @abc.abstractmethod
    def column_row_inequalities(
        self,
    ) -> tuple[set[tuple[Cell, Cell]], set[tuple[Cell, Cell]]]:
        """
        Return the inequalities for the row and column
        (this one checking that inequalities on the same row are strict).
        """


class LessThanRowColSeparation(AbstractSeparation):
    """
    When separating, cells must be strictly above/below each other.
    """

    def row_col_separation(self) -> Iterator[Tiling]:
        """
        Return the tiling with the row and column separated.
        """
        if any(self.tiling.find_empty_rows_and_columns()):
            yield self.tiling
            return
        row_col_map = self.row_col_map
        if row_col_map.is_identity():
            yield self.tiling
            return
        new_obstructions, new_requirements = row_col_map.preimage_of_tiling(self.tiling)
        new_dimensions = self.new_dimensions
        new_obstructions += self.new_obstructions
        for obs, reqs in self.point_row_obs_and_reqs():
            yield Tiling(
                new_obstructions + obs, new_requirements + reqs, new_dimensions
            )

    @property
    def new_active_cells(self) -> set[Cell]:
        """Return the new active cells of the tiling."""
        return set(self.map_cell(cell) for cell in self.tiling.active_cells)

    def point_row_obs_and_reqs(
        self,
    ) -> Iterator[tuple[Obstructions, Requirements]]:
        """
        Return the obstructions and requirements for the points in the rows.
        """
        yield (), ()

    @property
    def row_col_map(self) -> RowColMap:
        """Return the row and column map."""
        pre_row_indices = [next(iter(row_cell))[1] for row_cell in self.row_order]
        pre_col_indices = [next(iter(col_cell))[0] for col_cell in self.col_order]
        row_map = dict(enumerate(pre_row_indices))
        col_map = dict(enumerate(pre_col_indices))
        return RowColMap(col_map, row_map)

    def map_cell(self, cell: Cell) -> Cell:
        """
        Map the cell to its new position.
        """
        for idx, col in enumerate(self.col_order):
            if cell in col:
                for idx2, row in enumerate(self.row_order):
                    if cell in row:
                        return (idx, idx2)
        raise ValueError(f"Cell {cell} not found in the orders.")

    def column_row_inequalities(
        self,
    ) -> tuple[set[tuple[Cell, Cell]], set[tuple[Cell, Cell]]]:
        """
        Return the inequalities for the row and column
        (this one checking that inequalities on the same row are strict).
        """
        less_than_col, less_than_row, not_equal = self.inequalities_sets()
        return less_than_col, less_than_row.intersection(not_equal)


class LessThanOrEqualRowColSeparation(AbstractSeparation):
    """
    Allow cells to interleave in the top/bottom rows when
    separating cells in a row.
    """

    def row_col_separation(self) -> Iterator[Tiling]:
        """
        Return the tiling with the row and column separated.
        """
        if any(self.tiling.find_empty_rows_and_columns()):
            yield self.tiling
            return
        row_col_map = self.row_col_map
        if row_col_map.is_identity():
            yield self.tiling
            return
        new_obstructions, new_requirements = row_col_map.preimage_of_tiling(self.tiling)
        new_dimensions = self.new_dimensions
        new_obstructions += self.new_obstructions
        for obs, reqs in self.point_row_obs_and_reqs():
            yield Tiling(
                new_obstructions + obs, new_requirements + reqs, new_dimensions
            )

    def point_row_obs_and_reqs(
        self,
    ) -> Iterator[tuple[Obstructions, Requirements]]:
        """
        Return the obstructions and requirements for the points in the rows.
        """
        point_obs = self.point_obs()
        row_reqs: dict[int, Requirements] = {}
        row_obs: dict[int, Obstructions] = {}
        for row in self.point_rows:
            indices_of_above = []
            indices_of_below = []
            for cell in self.active_cells_in_row(row + 1):
                indices_of_above.append(cell[0])
            for cell in self.active_cells_in_row(row - 1):
                indices_of_below.append(cell[0])
            row_point_gcps_above = []
            row_point_gcps_below = []
            for i in indices_of_above:
                row_point_gcps_above.append(
                    GriddedCayleyPerm(CayleyPermutation([0]), ((i, row),))
                )
            for i in indices_of_below:
                row_point_gcps_below.append(
                    GriddedCayleyPerm(CayleyPermutation([0]), ((i, row),))
                )
            row_reqs[row] = (tuple(row_point_gcps_above), tuple(row_point_gcps_below))
            row_obs[row] = tuple(row_point_gcps_above + row_point_gcps_below)
        for i in range(len(self.point_rows) + 1):
            for positive_rows in combinations(self.point_rows, i):
                obs: list[GriddedCayleyPerm] = []
                reqs: list[tuple[GriddedCayleyPerm, ...]] = []
                for row in self.point_rows:
                    if row in positive_rows:
                        reqs.extend(row_reqs[row])
                    else:
                        obs.extend(row_obs[row])
                yield point_obs + tuple(obs), tuple(reqs)

    def point_obs(self) -> Obstructions:
        """Return the point obstructions."""
        point_obs = []
        for j in self.point_rows:
            cells = self.active_cells_in_row(j)
            for cell1, cell2 in combinations(sorted(cells), 2):
                point_obs.append(
                    GriddedCayleyPerm(CayleyPermutation([0, 1]), [cell1, cell2])
                )
                point_obs.append(
                    GriddedCayleyPerm(CayleyPermutation([1, 0]), [cell1, cell2])
                )
            for cell in cells:
                point_obs.append(
                    GriddedCayleyPerm(CayleyPermutation([0, 1]), [cell, cell])
                )
                point_obs.append(
                    GriddedCayleyPerm(CayleyPermutation([1, 0]), [cell, cell])
                )
        return tuple(point_obs)

    @property
    def new_active_cells(self) -> set[Cell]:
        """Return the new active cells of the tiling."""
        new_active_cells = [self.map_cell(cell) for cell in self.tiling.active_cells]
        point_row_active_cells = []
        for row in self.point_rows:
            for cell in new_active_cells:
                if cell[1] == row - 1 or cell[1] == row + 1:
                    point_row_active_cells.append((cell[0], row))
        return set(new_active_cells + point_row_active_cells)

    def point_row_cells(self, row: int) -> list[Cell]:
        """TODO: find the active cells of point rows, currently just adding them all."""
        point_row_cells = []
        i = self.new_dimensions[0]
        point_row_cells.append((i, row))
        return point_row_cells

    def active_cells_in_row(self, row: int) -> list[Cell]:
        """Returns the cells in the row of the separated tiling that are active."""
        return [cell for cell in self.new_active_cells if cell[1] == row]

    @property
    def row_col_map(self) -> RowColMap:
        """Return the row and column map."""
        pre_row_indices = [next(iter(row_cell))[1] for row_cell in self.row_order]
        pre_col_indices = [next(iter(col_cell))[0] for col_cell in self.col_order]
        row_map = {}
        prev = None
        count = 0
        for val in pre_row_indices:
            row_map[count] = val
            count += 1
            if val == prev:
                row_map[count] = val
                count += 1
            prev = val
        col_map = dict(enumerate(pre_col_indices))
        return RowColMap(col_map, row_map)

    @property
    def point_rows(self) -> list[int]:
        """Return the point rows."""
        point_rows: list[int] = []
        for i in range(self.tiling.dimensions[1]):
            point_rows.extend(
                self.row_col_map.preimages_of_row(i)[1::2]
            )  # finds the odd indices
        return point_rows

    def map_cell(self, cell: Cell) -> Cell:
        """
        Map the cell to its new position.
        """
        for idx, col in enumerate(self.col_order):
            if cell in col:
                count = 0
                previous = None
                for row in self.row_order:
                    val = next(iter(row))[1]
                    if val == previous:
                        count += 1
                    if cell in row:
                        return (idx, count)
                    count += 1
                    previous = val
        raise ValueError(f"Cell {cell} not found in the orders.")

    def column_row_inequalities(
        self,
    ) -> tuple[set[tuple[Cell, Cell]], set[tuple[Cell, Cell]]]:
        """
        Return the inequalities for the column and row
        (this one doesn't need to check if inequalities on the same row are strict).
        """
        less_than_col, less_than_row, _ = self.inequalities_sets()
        return less_than_col, less_than_row


class AbstractLessThanRowColSeparationStrategy(
    DisjointUnionStrategy[TilingT, GriddedCayleyPerm]
):
    """Abstract strategy for row and column separation."""

    # pylint: disable=duplicate-code
    def __init__(
        self,
        ignore_parent: bool = True,
        possibly_empty: bool = True,
    ):
        super().__init__(ignore_parent=ignore_parent, possibly_empty=possibly_empty)

    def formal_step(self) -> str:
        return "Separate rows and columns"

    @abc.abstractmethod
    def algorithm(self, comb_class: TilingT) -> AbstractSeparation:
        """The algorithm for finding the row and column separation."""

    def backward_map(
        self,
        comb_class: TilingT,
        objs: tuple[Optional[GriddedCayleyPerm], ...],
        children: Optional[tuple[TilingT, ...]] = None,
    ) -> Iterator[GriddedCayleyPerm]:
        rc_map = self.algorithm(comb_class).row_col_map
        for obj in objs:
            if obj is not None:
                yield rc_map.map_gridded_cperm(obj)

    def forward_map(
        self,
        comb_class: TilingT,
        obj: GriddedCayleyPerm,
        children: Optional[tuple[TilingT, ...]] = None,
    ) -> tuple[Optional[GriddedCayleyPerm], ...]:
        algo = self.algorithm(comb_class)
        rc_map = algo.row_col_map
        children_gcps = []
        for obs, reqs in algo.point_row_obs_and_reqs():
            for preimage in rc_map.preimage_of_gridded_cperm(obj):
                if (
                    preimage.avoids(algo.new_obstructions)
                    and preimage.avoids(obs)
                    and all(preimage.contains(req) for req in reqs)
                ):
                    children_gcps.append(preimage)
        assert len(children_gcps) == len(
            list(algo.point_row_obs_and_reqs())
        ), "More than one gcp mapped to the same child."
        return tuple(children_gcps)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ignore_parent={self.ignore_parent}, "
            f"possibly_empty={self.possibly_empty})"
        )

    def to_jsonable(self) -> dict:
        d: dict = super().to_jsonable()
        d.pop("workable")
        d.pop("inferrable")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractLessThanRowColSeparationStrategy":
        return cls(
            ignore_parent=d["ignore_parent"],
            possibly_empty=d["possibly_empty"],
        )


class LessThanRowColSeparationStrategy(
    AbstractLessThanRowColSeparationStrategy[Tiling]
):
    """A strategy that separates rows and columns."""

    def algorithm(self, comb_class: Tiling) -> LessThanRowColSeparation:
        """The algorithm for finding the row and column separation."""
        return LessThanRowColSeparation(comb_class)

    def decomposition_function(self, comb_class: Tiling) -> tuple[Tiling, ...]:
        """Return the decomposition function."""
        algo = self.algorithm(comb_class)
        if algo.row_col_map.is_identity():
            raise StrategyDoesNotApply
        return tuple(
            algo.row_col_separation(),
        )

    def extra_parameters(
        self, comb_class: Tiling, children: Optional[tuple[Tiling, ...]] = None
    ) -> tuple[dict[str, str], ...]:
        """Return the extra parameters for the strategy."""
        return tuple({} for _ in self.decomposition_function(comb_class))


class AbstractLessThanOrEqualRowColSeparationStrategy(
    AbstractLessThanRowColSeparationStrategy
):
    """A strategy that allows interleaving in the top/bottom rows when separating"""

    def __init__(
        self,
        row_order: list[set[Cell]],
        ignore_parent: bool = True,
        possibly_empty: bool = True,
    ):
        super().__init__(ignore_parent=ignore_parent, possibly_empty=possibly_empty)
        self.row_order = row_order

    def formal_step(self) -> str:
        """Return a string that describe the operation performed on the tiling."""
        return super().formal_step() + " allowing interleaving in top/bottom rows"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"row_order={self.row_order}, "
            f"ignore_parent={self.ignore_parent}, "
            f"possibly_empty={self.possibly_empty})"
        )

    def to_jsonable(self) -> dict:
        d: dict = super().to_jsonable()
        d["row_order"] = [list(row) for row in self.row_order]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractLessThanRowColSeparationStrategy":
        return cls(
            row_order=[set(tuple(cell) for cell in row) for row in d["row_order"]],
            ignore_parent=d["ignore_parent"],
            possibly_empty=d["possibly_empty"],
        )


class LessThanOrEqualRowColSeparationStrategy(
    AbstractLessThanOrEqualRowColSeparationStrategy
):
    """A strategy that allows interleaving in the top/bottom rows when separating"""

    def algorithm(self, comb_class):
        """The algorithm for finding the row and column separation."""
        return LessThanOrEqualRowColSeparation(comb_class, self.row_order)

    def decomposition_function(self, comb_class: Tiling) -> tuple[Tiling, ...]:
        """Return the decomposition function."""
        algo = self.algorithm(comb_class)
        if algo.row_col_map.is_identity():
            raise StrategyDoesNotApply
        return tuple(algo.row_col_separation())


class AbstractLessThanOrEqualRowColSeparationFactory(StrategyFactory[TilingT]):
    """A factory which returns separation strategies with
    all different row column separation strategies
    for a given tiling."""

    @abc.abstractmethod
    def algorithm(self, comb_class: TilingT) -> LessThanOrEqualRowColSeparation:
        """The algorithm for finding the row and column separation."""

    def row_separations(
        self,
        comb_class: TilingT,
    ) -> list[list[set[Cell]]]:
        """The possible separations of the rows of the tiling (for cols just use max_col_order)."""
        max_row_order = self.algorithm(comb_class).row_order
        row_separation_dict = defaultdict(list)
        for cells_set in max_row_order:
            row = list(cells_set)[0][1]
            row_separation_dict[row].append(cells_set)
        rows_exta_expanding = [
            row for row in row_separation_dict if len(row_separation_dict[row]) > 2
        ]
        if not rows_exta_expanding:
            return [max_row_order]
        corrected_row_orders = [max_row_order]
        return self.correct_row_orders(
            corrected_row_orders, rows_exta_expanding, row_separation_dict
        )

    def correct_row_orders(
        self,
        corrected_row_orders: list[list[set[Cell]]],
        rows_exta_expanding: list[int],
        row_separation_dict: dict[int, list[set[Cell]]],
    ) -> list[list[set[Cell]]]:
        """If any row separates into more than 2 rows then it merges them
        together in all possible ways so that they only separate into 2 rows.
        Returns all possible row orders obtained by doing this."""
        for row in rows_exta_expanding:
            to_merge = row_separation_dict[row]
            merged_cells = []
            for n in range(1, len(to_merge)):
                merge_left = set(
                    cell for set_of_cells in to_merge[:n] for cell in set_of_cells
                )
                merge_right = set(
                    cell for set_of_cells in to_merge[n:] for cell in set_of_cells
                )
                merged_cells.append((merge_left, merge_right))
            corrected_row_orders = self.update_old_orders(
                corrected_row_orders, to_merge, merged_cells
            )
        return corrected_row_orders

    def update_old_orders(
        self,
        old_corrected_row_orders: list[list[set[Cell]]],
        to_merge: list[set[Cell]],
        merged_cells: list[tuple[set[Cell], set[Cell]]],
    ) -> list[list[set[Cell]]]:
        """Updates the old row orders by replacing the cells in to_merge with
        the different merged cells in merged_cells."""
        new_corrected_row_orders = []
        for old_row_order in old_corrected_row_orders:
            left = []
            right = []
            passed_middle = False
            for cells_set in old_row_order:
                if cells_set in to_merge:
                    passed_middle = True
                elif not passed_middle:
                    left.append(cells_set)
                else:
                    right.append(cells_set)
            for merge_left, merge_right in merged_cells:
                new_corrected_row_orders.append(
                    left + [merge_left, merge_right] + right
                )
        return new_corrected_row_orders

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractLessThanOrEqualRowColSeparationFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return (
            "Separate rows and columns allowing interleaving in "
            "top/bottom rows in all different ways."
        )


class LessThanOrEqualRowColSeparationFactory(
    AbstractLessThanOrEqualRowColSeparationFactory
):
    """A factory which returns LessThanOrEqualRowColSeparationStrategys with
    all different row column separation strategies
    for a given tiling."""

    def algorithm(self, comb_class: Tiling) -> LessThanOrEqualRowColSeparation:
        """The algorithm for finding the row and column separation."""
        return LessThanOrEqualRowColSeparation(comb_class)

    def __call__(
        self, comb_class: Tiling
    ) -> Iterator[LessThanOrEqualRowColSeparationStrategy]:
        """Finds max expansion and if any row separates more than 2 cells then
        it merges them together so that each row splits into at most 2 rows
        (plus a point row between them) and yields all possible ways of doing this."""
        for row_order in self.row_separations(comb_class):
            yield LessThanOrEqualRowColSeparationStrategy(row_order=row_order)
