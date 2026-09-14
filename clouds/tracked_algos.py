"""The algorithms for tracked tilings."""

from functools import cached_property
from typing import Iterable, Optional
from gridded_cayley_permutations import GriddedCayleyPerm
from gridded_cayley_permutations.factors import Factors, ShuffleFactors
from gridded_cayley_permutations.point_placements import PointPlacement
from tilescope.strategies.row_column_separation import (
    LessThanRowColSeparation,
    LessThanOrEqualRowColSeparation,
)
from .tracked_tiling import TrackedTiling

Cell = tuple[int, int]


class TrackedFactors(Factors):
    """Factors and tracks clouds"""

    def __init__(self, tracked_tiling: TrackedTiling) -> None:
        self.tracked_tiling = tracked_tiling
        super().__init__(tracked_tiling.tiling)

    @cached_property
    def positive_point_rows_and_represantive(
        self,
    ) -> tuple[set[int], set[tuple[int, int]]]:
        """Return the positive point rows and their represantive cells."""
        positive_point_rows = set()
        positive_point_rows_reps = set()
        for point_row in self.tracked_tiling.point_rows:
            if any(point_row in cloud for cloud in self.tracked_tiling.value_clouds):
                positive_cells = [
                    cell
                    for cell in self.tracked_tiling.cells_in_row(point_row)
                    if cell in self.tracked_tiling.positive_cells()
                ]
                if positive_cells:
                    cell = min(positive_cells)
                    positive_point_rows.add(cell[1])
                    positive_point_rows_reps.add(cell)
        return positive_point_rows, positive_point_rows_reps

    def new_value_clouds(
        self, active_cells: set[tuple[int, int]]
    ) -> tuple[tuple[int, ...], ...]:
        """Return the new value clouds for a factor given its active cells."""
        (
            positive_point_rows,
            positive_point_rows_reps,
        ) = self.positive_point_rows_and_represantive
        return tuple(
            tuple(
                row
                for row in cloud
                if row not in positive_point_rows
                or positive_point_rows_reps.intersection(active_cells)
            )
            for cloud in self.tracked_tiling.value_clouds
        )

    def find_tracked_factors(self) -> Iterable[TrackedTiling]:
        """Return the factors of the tracked tiling."""
        factors = self.find_factors()
        # only one child where the row is positive needs a cloud for a point row
        for factor in factors:
            yield TrackedTiling(
                factor,
                value_clouds=self.new_value_clouds(factor.active_cells),
                indices_clouds=self.tracked_tiling.indices_clouds,
                intersect_clouds_with_active=True,
            )


class TrackedShuffleFactors(ShuffleFactors, TrackedFactors):
    """
    Return the interleaving factors, which drops the condition
    of cells being in the same row or column and trackes clouds.

    This is structurally sound, but no longer counted by Cartesian products.
    """


class TrackedLessThanRowColSeparation(LessThanRowColSeparation):
    """Separates rows and columns with less than constraints and tracks clouds."""

    def __init__(
        self, tracked_tiling: TrackedTiling, row_order: Optional[list[set[Cell]]] = None
    ) -> None:
        self.tracked_tiling = tracked_tiling
        super().__init__(tracked_tiling.tiling, row_order=row_order)

    def tracked_row_col_separation(self) -> Iterable[TrackedTiling]:
        """Yield the separated tilings with tracked clouds."""
        for separated_tiling in self.row_col_separation():
            indices_clouds, value_clouds = TrackedTiling.map_clouds(
                indices_clouds=self.tracked_tiling.indices_clouds,
                value_clouds=self.tracked_tiling.value_clouds,
                tiling_map=self.row_col_map,
            )
            yield TrackedTiling(
                separated_tiling,
                value_clouds=value_clouds,
                indices_clouds=indices_clouds,
                intersect_clouds_with_active=True,
            )


class TrackedLessThanOrEqualRowColSeparation(
    LessThanOrEqualRowColSeparation,
):
    """Separates rows and columns with less than or equal constraints and tracks clouds."""

    def __init__(
        self, tracked_tiling: TrackedTiling, row_order: Optional[list[set[Cell]]] = None
    ) -> None:
        self.tracked_tiling = tracked_tiling
        super().__init__(tracked_tiling.tiling, row_order=row_order)

    def tracked_row_col_separation(
        self,
    ) -> Iterable[TrackedTiling]:
        """Yield the separated tilings with tracked clouds."""
        for separated_tiling in self.row_col_separation():
            indices_clouds, value_clouds = TrackedTiling.map_clouds(
                indices_clouds=self.tracked_tiling.indices_clouds,
                value_clouds=self.tracked_tiling.value_clouds,
                tiling_map=self.row_col_map,
            )
            yield TrackedTiling(
                separated_tiling,
                value_clouds=value_clouds,
                indices_clouds=indices_clouds,
                intersect_clouds_with_active=True,
            )


class TrackedPointPlacement(PointPlacement):
    """Point placement tracking clouds."""

    def __init__(self, tracked_tiling: TrackedTiling) -> None:
        self.tracked_tiling = tracked_tiling
        super().__init__(tracked_tiling.tiling)

    def tracked_point_placement(
        self,
        requirement_list: tuple[GriddedCayleyPerm, ...],
        indices: tuple[int, ...],
        direction: int,
    ) -> tuple[TrackedTiling, ...]:
        """Yield the tilings with tracked clouds after point placement."""
        if direction not in self.DIRECTIONS:
            raise ValueError(f"Direction {direction} is not a valid direction.")
        cells = self.cells_to_place_in(requirement_list, indices)
        all_tracked_tilings = []
        for placed_cell in cells:
            tiling = self.point_placement_in_cell(
                requirement_list, indices, direction, placed_cell
            )
            map_for_cells = self.multiplex_map(placed_cell)
            indices_clouds, value_clouds = TrackedTiling.map_clouds(
                indices_clouds=self.tracked_tiling.indices_clouds,
                value_clouds=self.tracked_tiling.value_clouds,
                tiling_map=map_for_cells,
            )
            all_tracked_tilings.append(
                TrackedTiling(
                    tiling,
                    indices_clouds=indices_clouds,
                    value_clouds=value_clouds,
                    intersect_clouds_with_active=True,
                )
            )
        return tuple(all_tracked_tilings)

    def cells_to_place_in(
        self,
        requirement_list: tuple[GriddedCayleyPerm, ...],
        indices: tuple[int, ...],
    ) -> set[Cell]:
        """Return the set of cells to place points in."""
        cells = set()
        for idx, gcp in zip(indices, requirement_list):
            cells.add(gcp.positions[idx])
        return sorted(cells)
