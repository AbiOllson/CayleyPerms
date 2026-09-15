"""Strategies for row and column separation in tracked tilings."""

from typing import Iterator
from tilescope.strategies import (
    AbstractLessThanRowColSeparationStrategy,
    AbstractLessThanOrEqualRowColSeparationStrategy,
    AbstractLessThanOrEqualRowColSeparationFactory,
)
from .extra_parameters import ExtraParametersForStrategies
from ..tracked_tiling import TrackedTiling
from ..tracked_algos import (
    TrackedLessThanOrEqualRowColSeparation,
    TrackedLessThanRowColSeparation,
)

Cell = tuple[int, int]


class TrackedLessThanRowColSeparationStrategy(
    ExtraParametersForStrategies,
    AbstractLessThanRowColSeparationStrategy[TrackedTiling],
):
    """A strategy for separating rows and columns with less than constraints."""

    def algorithm(self, comb_class):
        """Return the algorithm for row and column separation."""
        return TrackedLessThanRowColSeparation(comb_class)

    def decomposition_function(
        self, comb_class: TrackedTiling
    ) -> tuple[TrackedTiling, ...]:
        """Return the decomposition function."""
        algo = self.algorithm(comb_class)
        return (next(algo.tracked_row_col_separation()),)

    def maps_for_clouds(self, comb_class: TrackedTiling):
        return (
            self.rc_map_for_cloud(self.algorithm(comb_class).row_col_map, comb_class),
        )


class TrackedLessThanOrEqualRowColSeparationStrategy(
    AbstractLessThanOrEqualRowColSeparationStrategy,
    TrackedLessThanRowColSeparationStrategy,
):
    # pylint: disable=too-many-ancestors
    """A strategy for separating rows and columns with less than or equal constraints."""

    def algorithm(self, comb_class):
        return TrackedLessThanOrEqualRowColSeparation(comb_class, self.row_order)

    def maps_for_clouds(self, comb_class: TrackedTiling):
        rc_map = self.rc_map_for_cloud(
            self.algorithm(comb_class).row_col_map, comb_class
        )
        all_maps = []
        for _ in self.decomposition_function(comb_class):
            all_maps.append(rc_map)
        return tuple(all_maps)

    def decomposition_function(
        self, comb_class: TrackedTiling
    ) -> tuple[TrackedTiling, ...]:
        """Return the decomposition function."""
        algo = self.algorithm(comb_class)
        return tuple(algo.tracked_row_col_separation())


class TrackedLessThanOrEqualRowColSeparationFactory(
    AbstractLessThanOrEqualRowColSeparationFactory
):
    """A factory for creating strategies for separating rows and columns
    with less than or equal constraints."""

    def algorithm(
        self, comb_class: TrackedTiling
    ) -> TrackedLessThanOrEqualRowColSeparation:
        """Return the algorithm for row and column separation."""
        return TrackedLessThanOrEqualRowColSeparation(comb_class)

    def __call__(
        self, comb_class: TrackedTiling
    ) -> Iterator[TrackedLessThanOrEqualRowColSeparationStrategy]:
        """Finds max expansion and if any row separates more than 2 cells then
        it merges them together so that each row splits into at most 2 rows
        (plus a point row between them) and yields all possible ways of doing this."""
        for row_order in self.row_separations(comb_class):
            yield TrackedLessThanOrEqualRowColSeparationStrategy(
                row_order=row_order,
            )
