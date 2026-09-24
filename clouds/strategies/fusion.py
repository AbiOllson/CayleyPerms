"""Tracked fusion strategies for tilings."""

from typing import Optional
from comb_spec_searcher.strategies.constructor import Constructor
from tilescope.strategies import (
    AbstractFusionFactory,
    AbstractFusionStrategy,
)
from ..tracked_tiling import TrackedTiling
from .extra_parameters import ExtraParametersForStrategies
from .fusion_constructor import (
    FusionConstructor,
    PointRowFusionConstructor,
    ReverseFusionConstructor,
)

Cell = tuple[int, int]

"""
DivideByN is from
https://github.com/PermutaTriangle/Tilings/blob/develop/tilings/strategies/unfusion.py

DivideByK is from
https://github.com/PermutaTriangle/Tilings/blob/develop/tilings/strategies/pointing.py

FusionRule is from
https://github.com/PermutaTriangle/Tilings/blob/develop/tilings/strategies/fusion/fusion.py
"""

from collections import defaultdict, Counter
from itertools import islice
from random import randint
from typing import Callable, Iterator, List, Optional, Tuple, cast, Dict
import sympy

from comb_spec_searcher.typing import (
    CombinatorialClassType,
    CombinatorialObjectType,
    Parameters,
    SubObjects,
    SubRecs,
    SubSamplers,
    SubTerms,
    Terms,
)
from comb_spec_searcher.strategies import NonBijectiveRule, DisjointUnion, Constructor
from comb_spec_searcher.typing import Objects
from comb_spec_searcher.exception import StrategyDoesNotApply
from gridded_cayley_permutations import GriddedCayleyPerm
from .fusion_constructor import FusionConstructor


class DivideByN(DisjointUnion[CombinatorialClassType, CombinatorialObjectType]):
    """
    A constructor that works as disjoint union
    but divides the values by n + shift.
    """

    def __init__(
        self,
        parent: CombinatorialClassType,
        children: Tuple[CombinatorialClassType, ...],
        shift: int,
        extra_parameters: Optional[Tuple[Dict[str, str], ...]] = None,
    ):
        self.shift = shift
        self.initial_conditions = {
            n: parent.get_terms(n) for n in range(1 - self.shift)
        }
        super().__init__(parent, children, extra_parameters)

    def get_equation(
        self, lhs_func: sympy.Function, rhs_funcs: Tuple[sympy.Function, ...]
    ) -> sympy.Eq:
        # TODO: d/dx [ x**shift * lhsfun ] / x**(shift - 1) = A + B + ...
        raise NotImplementedError

    def get_terms(
        self, parent_terms: Callable[[int], Terms], subterms: SubTerms, n: int
    ) -> Terms:
        if n + self.shift <= 0:
            return self.initial_conditions[n]
        terms = super().get_terms(parent_terms, subterms, n)
        return Counter({key: value // (n + self.shift) for key, value in terms.items()})

    def get_sub_objects(
        self, subobjs: SubObjects, n: int
    ) -> Iterator[
        Tuple[Parameters, Tuple[List[Optional[CombinatorialObjectType]], ...]]
    ]:
        raise NotImplementedError

    def random_sample_sub_objects(
        self,
        parent_count: int,
        subsamplers: SubSamplers,
        subrecs: SubRecs,
        n: int,
        **parameters: int,
    ) -> Tuple[Optional[CombinatorialObjectType], ...]:
        raise NotImplementedError

    @staticmethod
    def get_eq_symbol() -> str:
        return "?"

    def __str__(self):
        return "divide by n"

    def equiv(
        self, other: Constructor, data: Optional[object] = None
    ) -> Tuple[bool, Optional[object]]:
        raise NotImplementedError


class DivideByK(DivideByN):
    """
    A constructor that works as disjoint union
    but divides the values by k + shift.
    """

    def __init__(
        self,
        parent: CombinatorialClassType,
        children: Tuple[CombinatorialClassType, ...],
        shift: int,
        parameter: str,
        extra_parameters: Optional[Tuple[Dict[str, str], ...]] = None,
    ):
        self.parameter = parameter
        self.division_index = parent.extra_parameters.index(parameter)
        super().__init__(parent, children, shift, extra_parameters)

    def get_terms(
        self, parent_terms: Callable[[int], Terms], subterms: SubTerms, n: int
    ) -> Terms:
        if n + self.shift <= 0:
            return self.initial_conditions[n]
        terms = DisjointUnion.get_terms(self, parent_terms, subterms, n)
        return Counter(
            {
                key: (
                    value // (key[self.division_index] + self.shift)
                    if (key[self.division_index] + self.shift) != 0
                    else value
                )
                for key, value in terms.items()
            }
        )

    def __str__(self):
        return f"divide by {self.parameter}"


class FusionRule(NonBijectiveRule[TrackedTiling, GriddedCayleyPerm]):
    """Overwritten the generate objects of size method, as this relies on
    knowing the number of left and right points of the parent TrackedTiling."""

    @property
    def strategy(self) -> "TrackedFusionStrategy":
        return cast(
            TrackedFusionStrategy,
            super().strategy,
        )

    @property
    def constructor(self) -> FusionConstructor:
        return cast(FusionConstructor, super().constructor)

    def is_equivalence(
        self, is_empty: Optional[Callable[[TrackedTiling], bool]] = None
    ) -> bool:
        return False

    def _ensure_level_objects(self, n: int) -> None:
        if self.subobjects is None:
            raise RuntimeError("set_subrecs must be set first")
        while n >= len(self.objects_cache):
            res: Objects = defaultdict(list)
            min_left, min_right = self.constructor.min_points

            def add_new_gp(
                params: List[int],
                left_points: int,
                fuse_region_points: int,
                unfused_gps: Iterator[GriddedCayleyPerm],
            ) -> None:
                """Update new terms if there is enough points on the left and right."""
                gp = next(unfused_gps)
                if (
                    min_left <= left_points
                    and min_right <= fuse_region_points - left_points
                ):
                    res[tuple(params)].append(gp)

            for param, objects in self.subobjects[0](len(self.objects_cache)).items():
                fuse_region_points = param[self.constructor.fuse_parameter_index]
                for gp in objects:
                    new_params = list(self.constructor.children_param_map(param))
                    unfused_gps = self.strategy.backward_map(
                        self.comb_class, (gp,), self.children
                    )  # iterates over unfused gridded perms in order
                    # with 0, 1, .., and finally fuse_region_points on the left
                    for idx in self.constructor.left_parameter_indices:
                        new_params[idx] -= fuse_region_points
                    add_new_gp(new_params, 0, fuse_region_points, unfused_gps)
                    for left_points in range(1, fuse_region_points + 1):
                        for idx in self.constructor.left_parameter_indices:
                            new_params[idx] += 1
                        for idx in self.constructor.right_parameter_indices:
                            new_params[idx] -= 1
                        add_new_gp(
                            new_params, left_points, fuse_region_points, unfused_gps
                        )

            self.objects_cache.append(res)

    def random_sample_object_of_size(
        self, n: int, **parameters: int
    ) -> GriddedCayleyPerm:
        """Return a random objects of the give size."""
        assert (
            self.subrecs is not None and self.subsamplers is not None
        ), "you must call the set_subrecs function first"
        subrec = self.subrecs[0]
        subsampler = self.subsamplers[0]
        parent_count = self.count_objects_of_size(n, **parameters)
        random_choice = randint(1, parent_count)
        total = 0
        left_right_points = self.constructor.determine_number_of_points_in_fuse_region(
            n, **parameters
        )
        for left_points, right_points in left_right_points:
            new_params = self.constructor.update_subparams(
                left_points, right_points, **parameters
            )
            if new_params is not None:
                assert (
                    new_params[self.constructor.fuse_parameter]
                    == left_points + right_points
                )
                total += subrec(n, **new_params)
                if random_choice <= total:
                    gp = subsampler(n, **new_params)
                    try:
                        return next(
                            self.strategy.backward_map(
                                self.comb_class, (gp,), self.children, left_points
                            )
                        )
                    except StopIteration:
                        assert 0, "something went wrong"
        raise RuntimeError("The for-loop for randomly sampling objects was empty")

    def _forward_order(
        self,
        obj: GriddedCayleyPerm,
        image: Tuple[Optional[GriddedCayleyPerm], ...],
        data: Optional[object] = None,
    ) -> int:
        # The position of the original object in the backward map of the child object
        return next(i for i, gp in enumerate(self.backward_map(image)) if gp == obj)

    def _backward_order_item(
        self,
        idx: int,
        objs: Tuple[Optional[GriddedCayleyPerm], ...],
        data: Optional[object] = None,
    ) -> GriddedCayleyPerm:
        if data:  # reverse order
            return tuple(self.backward_map(objs))[-idx - 1]
        return next(islice(self.backward_map(objs), idx, None))


class AbstractTrackedFusionStrategy(
    ExtraParametersForStrategies, AbstractFusionStrategy[TrackedTiling]
):
    """Abstract fusion strategy for tracked tilings."""

    def maps_for_clouds(self, comb_class: TrackedTiling):
        rc_map = self.fusion_map(comb_class)
        clouds_col_map = {
            x: (rc_map.col_map[x],) for x in range(comb_class.dimensions[0])
        }
        clouds_row_map = {
            x: (rc_map.row_map[x],) for x in range(comb_class.dimensions[1])
        }
        return ((clouds_col_map, clouds_row_map),)

    def sided_parameters(self, comb_class: TrackedTiling):
        """Determine which parameters are left-sided, right-sided, or both-sided."""
        left_sided_parameters = []
        right_sided_parameters = []
        both_sided_parameters = []
        if self.fuse_rows:
            all_clouds = comb_class.value_clouds
        else:
            all_clouds = comb_class.indices_clouds

        for cloud in all_clouds:
            intersects_left = any(idx == self.index for idx in cloud)
            intersects_right = any(idx == self.index + 1 for idx in cloud)
            if intersects_left and intersects_right:
                both_sided_parameters.append(
                    comb_class.find_parameter(cloud, self.fuse_rows)
                )
            elif intersects_left:
                left_sided_parameters.append(
                    comb_class.find_parameter(cloud, self.fuse_rows)
                )
            elif intersects_right:
                right_sided_parameters.append(
                    comb_class.find_parameter(cloud, self.fuse_rows)
                )
        return left_sided_parameters, right_sided_parameters, both_sided_parameters

    def __call__(
        self,
        comb_class: TrackedTiling,
        children: Optional[Tuple[TrackedTiling, ...]] = None,
    ) -> FusionRule:
        if children is None:
            children = self.decomposition_function(comb_class)
            if children is None:
                raise StrategyDoesNotApply("Strategy does not apply")
        return FusionRule(self, comb_class, children=children)


class TrackedFusionStrategy(
    AbstractTrackedFusionStrategy,
):
    """Tracked fusion strategy."""

    def __init__(self, fuse_rows: bool, index: int, tracked: bool = True):
        super().__init__(fuse_rows=fuse_rows, index=index, tracked=tracked)

    def decomposition_function(self, comb_class: TrackedTiling) -> tuple[TrackedTiling]:
        return (comb_class.fuse(self.fuse_rows, self.index),)

    def constructor(
        self,
        comb_class: TrackedTiling,
        children: Optional[tuple[TrackedTiling, ...]] = None,
    ) -> FusionConstructor:
        """
        This is where the details of the 'reliance profile' and 'counting'
        functions are hidden.
        """
        if children is None:
            children = self.decomposition_function(comb_class)
        child = children[0]
        fuse_parameter = child.find_parameter((self.index,), self.fuse_rows)
        extra_parameters = self.extra_parameters(comb_class, children)
        (
            left_sided_parameters,
            right_sided_parameters,
            both_sided_parameters,
        ) = self.sided_parameters(comb_class)
        return FusionConstructor(
            comb_class,
            child,
            fuse_parameter,
            extra_parameters[0],
            left_sided_parameters,
            right_sided_parameters,
            both_sided_parameters,
            0,
            0,
        )

    def reverse_constructor(self, idx, comb_class, children=None):
        if children is None:
            children = self.decomposition_function(comb_class)
        child = children[0]
        fuse_parameter = child.find_parameter((self.index,), self.fuse_rows)
        extra_parameters = self.extra_parameters(comb_class, children)
        left_sided_parameters, right_sided_parameters, _ = self.sided_parameters(
            comb_class
        )
        return ReverseFusionConstructor(
            comb_class,
            child,
            fuse_parameter,
            extra_parameters[0],
            left_sided_parameters,
            right_sided_parameters,
            0,
            0,
        )

    def is_reversible(self, comb_class: TrackedTiling):
        row_map, col_map = self.maps_for_clouds(comb_class)[0]
        child = self.decomposition_function(comb_class)[0]
        fuse_cloud = (self.index,)
        if self.fuse_rows:
            value_clouds = [
                self.map_cloud(cloud, row_map, child, rows=True)
                for cloud in comb_class.value_clouds
            ]
            return fuse_cloud in value_clouds
        index_clouds = [
            self.map_cloud(cloud, col_map, child, rows=False)
            for cloud in comb_class.indices_clouds
        ]
        return fuse_cloud in index_clouds


class TrackedFusionFactory(AbstractFusionFactory):
    """Factory for doing fusion."""

    def __call__(self, comb_class: TrackedTiling):
        for direction in [True, False]:
            for index in range(comb_class.dimensions[direction] - 1):
                if comb_class.is_fusable(direction, index):
                    yield TrackedFusionStrategy(direction, index)


class TrackedFusionPointRowStrategy(AbstractTrackedFusionStrategy):
    """Tracked point row fusion strategy for fusing together rows,
    at least one of which is a point row."""

    def __init__(self, fuse_rows: bool, index: int, tracked: bool = True):
        super().__init__(fuse_rows=fuse_rows, index=index, tracked=tracked)

    def decomposition_function(self, comb_class: TrackedTiling) -> tuple[TrackedTiling]:
        """If self.index is a point row then remove it, otherwise self.index + 1 is a point row so
        remove that."""
        return (comb_class.fuse(True, self.index),)

    def formal_step(self) -> str:
        idx = self.index
        return f"Point row fusion of row {idx} and row {idx + 1}"

    def constructor(
        self,
        comb_class: TrackedTiling,
        children: Optional[tuple[TrackedTiling, ...]] = None,
    ) -> PointRowFusionConstructor:
        """
        This is where the details of the 'reliance profile' and 'counting'
        functions are hidden.
        """
        if children is None:
            children = self.decomposition_function(comb_class)
        child = children[0]
        fuse_parameter = child.find_parameter((self.index,), self.fuse_rows)
        extra_parameters = self.extra_parameters(comb_class, children)
        left_sided_parameters, right_sided_parameters, _ = self.sided_parameters(
            comb_class
        )
        above = self.index + 1 in comb_class.point_rows
        return PointRowFusionConstructor(
            comb_class,
            child,
            fuse_parameter,
            extra_parameters[0],
            left_sided_parameters,
            right_sided_parameters,
            above,
        )

    def reverse_constructor(
        self,
        idx: int,
        comb_class: TrackedTiling,
        children: Optional[tuple[TrackedTiling, ...]] = None,
    ) -> Constructor:
        raise NotImplementedError


class TrackedFusionPointRowFactory(AbstractFusionFactory):
    """Factory for fusing point rows/columns in tracked tilings."""

    def __call__(self, comb_class: TrackedTiling):
        for row in range(comb_class.dimensions[1] - 1):
            if comb_class.is_point_row_fuseable(row):
                yield TrackedFusionPointRowStrategy(True, row)

    def __str__(self) -> str:
        return "Fusion for point rows factory"
