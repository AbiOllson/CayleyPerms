"""Module for fusion strategy and factory"""

from typing import Tuple, Optional, Dict, Set, Iterator
from comb_spec_searcher import StrategyFactory, Strategy
from comb_spec_searcher.exception import StrategyDoesNotApply
from comb_spec_searcher.strategies.constructor import Constructor
from gridded_cayley_permutations import Tiling, GriddedCayleyPerm, RowColMap
from gridded_cayley_permutations.point_placements import TilingT


class AbstractFusionStrategy(Strategy[TilingT, GriddedCayleyPerm]):
    """Abstract strategy for fusing two rows or columns of a tiling together."""

    def __init__(self, fuse_rows: bool, index: int, tracked: bool = False):
        self.fuse_rows = fuse_rows
        self.index = index
        self.tracked = tracked
        if index < 0:
            raise ValueError("Index must be non-negative")
        super().__init__(
            ignore_parent=False, inferrable=True, possibly_empty=False, workable=True
        )

    def can_be_equivalent(self) -> bool:
        return False

    def is_two_way(self, comb_class: TilingT) -> bool:
        return False

    def is_reversible(self, comb_class: TilingT) -> bool:
        """TODO: We told this to return true to make it work but
        for tracked tilings and counting will need to change"""
        return True

    def shifts(
        self, comb_class: TilingT, children: Optional[Tuple[TilingT, ...]] = None
    ) -> Tuple[int, ...]:
        return (0,)

    def left_right_both_sided_parameters(
        self, comb_class: TilingT
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """Returns parameters."""
        raise NotImplementedError

    def _fuse_parameter(self, comb_class: TilingT) -> str:
        raise NotImplementedError

    def formal_step(self) -> str:
        fusing = "rows" if self.fuse_rows else "columns"
        idx = self.index
        return f"Fuse {fusing} {idx} and {idx + 1}"

    # pylint: disable=arguments-differ
    def backward_map(
        self,
        comb_class: TilingT,
        objs: Tuple[Optional[GriddedCayleyPerm], ...],
        children: Optional[Tuple[TilingT, ...]] = None,
        left_points: Optional[int] = None,
    ) -> Iterator[GriddedCayleyPerm]:
        """
        The backward direction of the underlying bijection used for object
        generation and sampling.
        """
        for obj in objs:
            if obj is not None:
                for preimage in self.fusion_map(comb_class).preimage_of_gridded_cperm(
                    obj
                ):
                    if left_points is None:
                        yield preimage
                    else:
                        if (
                            self.fuse_rows
                            and len(set(preimage.values_in_row(self.index)))
                            == left_points
                        ):
                            yield preimage
                        elif (
                            not self.fuse_rows
                            and len(preimage.indices_in_col(self.index)) == left_points
                        ):
                            yield preimage

    def forward_map(
        self,
        comb_class: TilingT,
        obj: GriddedCayleyPerm,
        children: Optional[Tuple[TilingT, ...]] = None,
    ) -> Tuple[Optional[GriddedCayleyPerm], ...]:
        """
        The forward direction of the underlying bijection used for object
        generation and sampling.
        """
        return (self.fusion_map(comb_class).map_gridded_cperm(obj),)

    def fusion_map(self, comb_class: TilingT):
        """Returns the rc map from parent to child."""
        if self.fuse_rows:
            col_map = {x: x for x in range(comb_class.dimensions[0])}
            row_map = {
                x: x if x <= self.index else x - 1
                for x in range(comb_class.dimensions[1])
            }
        else:
            row_map = {x: x for x in range(comb_class.dimensions[1])}
            col_map = {
                x: x if x <= self.index else x - 1
                for x in range(comb_class.dimensions[0])
            }
        return RowColMap(col_map, row_map)

    def to_jsonable(self) -> dict:
        d = super().to_jsonable()
        d.pop("ignore_parent")
        d.pop("inferrable")
        d.pop("possibly_empty")
        d.pop("workable")
        d["fuse_rows"] = self.fuse_rows
        d["index"] = self.index
        d["tracked"] = self.tracked
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractFusionStrategy":
        return cls(**d)

    @staticmethod
    def get_op_symbol() -> str:
        return "⚮"

    @staticmethod
    def get_eq_symbol() -> str:
        return "↣"

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(fuse_rows={self.fuse_rows}, index={self.index}, "
            f"tracked={self.tracked})"
        )


class FusionStrategy(AbstractFusionStrategy[Tiling]):
    """Strategy that fuses two rows or columns of a tiling together."""

    def decomposition_function(self, comb_class: Tiling) -> tuple[Tiling]:
        return (comb_class.fuse(self.fuse_rows, self.index),)

    def constructor(
        self, comb_class: Tiling, children: Optional[Tuple[Tiling, ...]] = None
    ):
        if not self.tracked:
            # constructor only enumerates when tracked.
            raise NotImplementedError("The fusion strategy was not tracked.")
        # Need to recompute some info to count, so ignoring passed in children
        if not comb_class.is_fusable(self.fuse_rows, self.index):
            raise StrategyDoesNotApply("Strategy does not apply")
        child = comb_class.fuse(self.fuse_rows, self.index)
        assert children is None or children == (child,)
        raise NotImplementedError

    def reverse_constructor(
        self,
        idx: int,
        comb_class: Tiling,
        children: Optional[Tuple[Tiling, ...]] = None,
    ) -> Constructor:
        raise NotImplementedError

    def extra_parameters(
        self, comb_class: Tiling, children: Optional[Tuple[Tiling, ...]] = None
    ) -> Tuple[Dict[str, str]]:
        raise NotImplementedError


class AbstractFusionFactory(StrategyFactory[TilingT]):
    """Abstract factory for doing fusion."""

    @classmethod
    def from_dict(cls, d: dict) -> "AbstractFusionFactory":
        return cls(**d)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return "Fusion factory"


class FusionFactory(AbstractFusionFactory[Tiling]):
    """Factory for doing fusion."""

    def __call__(self, comb_class: Tiling):
        for direction in [True, False]:
            for index in range(comb_class.dimensions[direction] - 1):
                if comb_class.is_fusable(direction, index):
                    yield FusionStrategy(direction, index)


class FusionPointRowStrategy(AbstractFusionStrategy):
    """Strategy that fuses a point row with an adjacent row of a tiling together."""

    def decomposition_function(self, comb_class: Tiling) -> tuple[Tiling]:
        """If self.index is a point row then remove it, otherwise self.index + 1 is a point row so
        remove that."""
        return (comb_class.fuse(True, self.index),)

    def formal_step(self) -> str:
        idx = self.index
        return f"Point row fusion of row {idx} and row {idx + 1}"

    def constructor(
        self, comb_class: Tiling, children: Optional[Tuple[Tiling, ...]] = None
    ):
        if not self.tracked:
            # constructor only enumerates when tracked.
            raise NotImplementedError("The fusion strategy was not tracked.")
        # Need to recompute some info to count, so ignoring passed in children
        if not comb_class.is_fusable(self.fuse_rows, self.index):
            raise StrategyDoesNotApply("Strategy does not apply")
        child = comb_class.fuse(self.fuse_rows, self.index)
        assert children is None or children == (child,)
        raise NotImplementedError

    def reverse_constructor(
        self,
        idx: int,
        comb_class: Tiling,
        children: Optional[Tuple[Tiling, ...]] = None,
    ) -> Constructor:
        raise NotImplementedError


class FusionPointRowFactory(AbstractFusionFactory):
    """Factory for doing fusion with point rows."""

    def __call__(self, comb_class: Tiling):
        for row in range(comb_class.dimensions[1] - 1):
            if comb_class.is_point_row_fuseable(row):
                yield FusionPointRowStrategy(True, row)

    def __str__(self) -> str:
        return "Fusion for point rows factory"
