"""Places a point requirement in a tiling in extreme directions.
0 = rightmost
1 = topmost, taking the rightmost if there are multiple
2 = topmost, taking the leftmost if there are multiple
3 = leftmost
4 = bottommost, taking the leftmost if there are multiple
5 = bottommost, taking the rightmost if there are multiple"""

import abc
from typing import Dict, Iterable, Iterator, Optional, Tuple
from comb_spec_searcher import DisjointUnionStrategy, StrategyFactory
from comb_spec_searcher.exception import StrategyDoesNotApply

from gridded_cayley_permutations import Tiling
from gridded_cayley_permutations.point_placements import (
    PointPlacement,
    AbstractPointPlacement,
    DIRECTIONS,
    DIR_RIGHT,
    DIR_RIGHT_TOP,
    DIR_LEFT_TOP,
    DIR_LEFT,
    DIR_LEFT_BOT,
    DIR_RIGHT_BOT,
    TilingT,
)
from gridded_cayley_permutations import GriddedCayleyPerm
from cayley_permutations import CayleyPermutation

Cell = Tuple[int, int]


class AbstractRequirementPlacementStrategy(
    DisjointUnionStrategy[TilingT, GriddedCayleyPerm]
):
    """Abstract strategy for placing a point of a requirement in a tiling in a direction.

    0 = rightmost
    1 = top rightmost
    2 = top leftmost
    3 = leftmost
    4 = bottom leftmost
    5 = bottom rightmost"""

    DIRECTIONS = DIRECTIONS

    def __init__(
        self,
        gcps: Iterable[GriddedCayleyPerm],
        indices: Iterable[int],
        direction: int,
        ignore_parent: bool = False,
    ):
        self.gcps = tuple(gcps)
        self.indices = tuple(indices)
        self.direction = direction
        assert direction in self.DIRECTIONS
        super().__init__(ignore_parent=ignore_parent)

    @abc.abstractmethod
    def algorithm(self, tiling: TilingT) -> AbstractPointPlacement:
        """Return the algorithm to be used for point placement."""

    def extra_parameters(
        self, comb_class: TilingT, children: Optional[Tuple[TilingT, ...]] = None
    ) -> Tuple[Dict[str, str], ...]:
        out = self.decomposition_function(comb_class)
        if out is None:
            raise StrategyDoesNotApply("Strategy does not apply")
        return tuple({} for _ in out)

    def formal_step(self):
        return (
            f"Placed the point of the requirement {self.gcps} "
            + f"at indices {self.indices} in direction {self.direction}"
        )

    def backward_map(
        self,
        comb_class: TilingT,
        objs: Tuple[Optional[GriddedCayleyPerm], ...],
        children: Optional[Tuple[TilingT, ...]] = None,
    ) -> Iterator[GriddedCayleyPerm]:
        placed_cells = sorted(
            set([gcp.positions[i] for gcp, i in zip(self.gcps, self.indices)])
        )
        placed_cells = [(x + 1, y + 1) for x, y in placed_cells]
        for placed_cell, obj in zip(placed_cells, objs[1:]):
            if obj is None:
                continue
            new_positions = []
            for cell in obj.positions:
                adjust_cols = 2 if placed_cell[0] < cell[0] else 0
                if cell[0] == placed_cell[0]:
                    adjust_cols += 1
                adjust_rows = 2 if placed_cell[1] < cell[1] else 0
                if cell[1] == placed_cell[1]:
                    adjust_rows += 1
                new_cell = (cell[0] - adjust_cols, cell[1] - adjust_rows)
                new_positions.append(new_cell)
            yield GriddedCayleyPerm(obj.pattern, tuple(new_positions))

    def forward_map(
        self,
        comb_class: TilingT,
        obj: GriddedCayleyPerm,
        children: Optional[Tuple[TilingT, ...]] = None,
    ) -> Tuple[Optional[GriddedCayleyPerm], ...]:
        cells = [gp.positions[idx] for gp, idx in zip(self.gcps, self.indices)]
        if obj.avoids(self.gcps):
            return (obj,) + tuple(None for _ in cells)
        # find the forced point, idx, val in obj
        forced_point = self.forced_point(obj)
        if forced_point is None:
            return (obj,) + tuple(None for _ in cells)
        # determine which child place the cell obj.positions[idx]
        cell = obj.positions[forced_point[0]]
        child_idx = sorted(cells).index(cell)
        new_positions = []
        for (idx, val), (x, y) in zip(enumerate(obj.pattern), obj.positions):
            if idx == forced_point[0]:
                x += 1
            if idx > forced_point[0]:
                x += 2
            if val == forced_point[1]:
                y += 1
            if val > forced_point[1]:
                y += 2
            new_positions.append((x, y))

        res = (None,) + tuple(
            GriddedCayleyPerm(obj.pattern, new_positions) if child_idx == idx else None
            for idx in range(len(cells))
        )
        return res

    def forced_point(self, gcp: GriddedCayleyPerm) -> Tuple[int, int] | None:
        """find the forced point"""
        point = None
        for idx, patt in zip(self.indices, self.gcps):
            for occ in patt.occurrences_in(gcp):
                new_point = occ[idx], gcp.pattern[occ[idx]]
                if point is None or PointPlacement.farther(
                    new_point, point, self.direction
                ):
                    point = new_point
        return point

    def __repr__(self) -> str:
        return (
            f"RequirementPlacementStrategy(gcps={self.gcps}, "
            f"indices={self.indices}, direction={self.direction}, "
            f"ignore_parent={self.ignore_parent})"
        )

    def to_jsonable(self) -> dict:
        """Return a dictionary form of the strategy."""
        d: dict = super().to_jsonable()
        d.pop("workable")
        d.pop("inferrable")
        d.pop("possibly_empty")
        d["gcps"] = tuple(gp.to_jsonable() for gp in self.gcps)
        d["indices"] = self.indices
        d["direction"] = self.direction
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractRequirementPlacementStrategy":
        """Return a strategy from a dictionary."""
        gcps = tuple(GriddedCayleyPerm.from_dict(gcp) for gcp in d.pop("gcps"))
        return cls(gcps=gcps, **d)


class RequirementPlacementStrategy(AbstractRequirementPlacementStrategy[Tiling]):
    """Insert a point of a requirement into a tiling in a direction."""

    def algorithm(self, tiling: Tiling) -> PointPlacement:  #
        """Return the point placement algorithm to use for the strategy."""
        return PointPlacement(tiling)

    def decomposition_function(self, comb_class: Tiling) -> Tuple[Tiling, ...]:
        return (comb_class.add_obstructions(self.gcps),) + self.algorithm(
            comb_class
        ).point_placement(self.gcps, self.indices, self.direction)


class VerticalInsertionEncodingPlacementFactory(StrategyFactory[Tiling]):
    """Factory for creating RequirementPlacementStrategy to place points
    for the vertical insertion encoding."""

    def __call__(self, comb_class: Tiling) -> Iterator[RequirementPlacementStrategy]:
        cells = comb_class.active_cells
        gcps = tuple(
            GriddedCayleyPerm(CayleyPermutation([0]), [cell]) for cell in cells
        )
        indices = tuple(0 for _ in gcps)
        direction = DIR_LEFT_BOT
        yield RequirementPlacementStrategy(gcps, indices, direction)

    @classmethod
    def from_dict(cls, d: dict) -> "VerticalInsertionEncodingPlacementFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Place next point of insertion encoding"


class HorizontalInsertionEncodingPlacementFactory(StrategyFactory[Tiling]):
    """Factory for creating RequirementPlacementStrategy to place points
    for the horizontal insertion encoding."""

    def __call__(self, comb_class: Tiling) -> Iterator[RequirementPlacementStrategy]:
        cells = comb_class.active_cells
        gcps = tuple(
            GriddedCayleyPerm(CayleyPermutation([0]), [cell]) for cell in cells
        )
        indices = tuple(0 for _ in gcps)
        direction = DIR_LEFT
        yield RequirementPlacementStrategy(gcps, indices, direction)

    @classmethod
    def from_dict(cls, d: dict) -> "HorizontalInsertionEncodingPlacementFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Place next point of insertion encoding"


class AbstractPointPlacementFactory(StrategyFactory[TilingT]):
    """Abstract factory for creating RequirementPlacementStrategy to place points."""

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractPointPlacementFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Point placement"


class PointPlacementFactory(AbstractPointPlacementFactory[Tiling]):
    """Factory for doing point placements."""

    def __call__(self, comb_class: Tiling) -> Iterator[RequirementPlacementStrategy]:
        for cell in comb_class.positive_cells():
            for direction in DIRECTIONS:
                gcps = (GriddedCayleyPerm(CayleyPermutation([0]), (cell,)),)
                indices = (0,)
                yield RequirementPlacementStrategy(gcps, indices, direction)


class AbstractRowInsertionFactory(StrategyFactory[TilingT]):
    """Abstract actory for having a point requirement on a row."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Row insertion"

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractRowInsertionFactory":
        return cls(**d)


class RowInsertionFactory(AbstractRowInsertionFactory[Tiling]):
    """Factory for having a point requirement on a row."""

    def __call__(self, comb_class: Tiling) -> Iterator[RequirementPlacementStrategy]:
        not_point_rows = set(range(comb_class.dimensions[1])) - comb_class.point_rows
        for row in not_point_rows:
            all_gcps = []
            for col in range(comb_class.dimensions[0]):
                cell = (col, row)
                gcps = GriddedCayleyPerm(CayleyPermutation([0]), (cell,))
                all_gcps.append(gcps)
            indices = tuple(0 for _ in all_gcps)
            for direction in [DIR_LEFT_BOT, DIR_RIGHT_BOT, DIR_LEFT_TOP, DIR_RIGHT_TOP]:
                yield RequirementPlacementStrategy(all_gcps, indices, direction)


class AbstractColInsertionFactory(StrategyFactory[TilingT]):
    """Abstract factory for having a point requirement on a column."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Column insertion"

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractColInsertionFactory":
        return cls(**d)


class ColInsertionFactory(AbstractColInsertionFactory[Tiling]):
    """Factory for having a point requirement on a column."""

    def __call__(self, comb_class: Tiling) -> Iterator[RequirementPlacementStrategy]:
        not_point_cols = set(range(comb_class.dimensions[0])) - set(
            cell[0] for cell in comb_class.point_cells()
        )
        for col in not_point_cols:
            all_gcps = []
            for row in range(comb_class.dimensions[1]):
                cell = (col, row)
                gcps = GriddedCayleyPerm(CayleyPermutation([0]), (cell,))
                all_gcps.append(gcps)
            indices = tuple(0 for _ in all_gcps)
            for direction in [DIR_LEFT, DIR_RIGHT]:
                yield RequirementPlacementStrategy(all_gcps, indices, direction)
