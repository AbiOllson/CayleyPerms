"""Module containing various strategy packs for running TileScope
in an easy to use format."""

from typing import Iterable
from comb_spec_searcher import (
    StrategyFactory,
    StrategyPack,
    AtomStrategy,
    VerificationStrategy,
)
from gridded_cayley_permutations import Tiling
from tilescope.strategies import (
    RemoveEmptyRowsAndColumnsStrategy,
    FactorStrategy,
    ShuffleFactorStrategy,
    VerticalInsertionEncodingPlacementFactory,
    VerticalInsertionEncodingRequirementInsertionFactory,
    HorizontalInsertionEncodingPlacementFactory,
    HorizontalInsertionEncodingRequirementInsertionFactory,
    CellInsertionFactory,
    PointPlacementFactory,
    LessThanRowColSeparationStrategy,
    LessThanOrEqualRowColSeparationFactory,
    RowInsertionFactory,
    ColInsertionFactory,
    VerticalInsertionEncodableVerificationStrategy,
    HorizontalInsertionEncodableVerificationStrategy,
    SubclassVerificationStrategy,
    FusionPointRowFactory,
    FusionFactory,
    ObstructionTransitivityStrategy,
)


class TileScopePack(StrategyPack):
    """Strategy packs for TileScope."""

    # pylint:disable=too-many-public-methods

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # General pack creation methods

    @classmethod
    def make_pack(
        cls,
        expansions: Iterable[str],
        cell_insertion: str = "none",
        point_placement_initial: bool = False,
        verify: bool | list[str] = False,
        root: Tiling | None = None,
        fusion: int = 0,
        shuffle_factors: bool = False,
    ):
        """Make a strategy pack with the given strategies.
        expansions: The expansion strategies to use; a list of strings where
            - "point": PointPlacementFactory
            - "row": RowInsertionFactory
            - "col": ColInsertionFactory

        cell_insertion: Where to include CellInsertionFactory. Must be included for point placement.
            - "initial": included as an initial strategy.
            - "expansion": included as an expansion strategy.
            - "none": not included.

        point_placement_initial: If True, includes PointPlacementFactory as an initial strategy.

        verify: If True then includes all verification strategies, otherwise just AtomStrategy.
        The root tiling must also be provided for subclass verification.
        Or input to verify can be a list of strings specifying which verification strategies to add.
            - "vert_ins_enc": VerticalInsertionEncodableVerificationStrategy
            - "hori_ins_enc": HorizontalInsertionEncodableVerificationStrategy
            - "subclass": SubclassVerificationStrategy

        root: The root tiling. Only needed if verify is True.

        fusion: If 0, no fusion. If 1, standard fusion. If 2, point row fusion. If 3, both.

        shuffle_factors: If True, replaces FactorStrategy with ShuffleFactorStrategy.
        """
        # pylint:disable=too-many-arguments
        # pylint:disable=too-many-positional-arguments
        # pylint:disable=too-many-branches
        if not expansions and cell_insertion != "expansion":
            raise ValueError("At least one expansion strategy must be provided.")
        pack = cls.basics()

        exp_strats: list[StrategyFactory] = []
        exp_name = "_and_".join(expansions) + "_placement" if expansions else ""
        for strat in expansions:
            if strat == "point":
                if cell_insertion == "none":
                    raise ValueError(
                        "Cell insertion must be included for point placement."
                    )
                if point_placement_initial:
                    raise ValueError(
                        "Point placement cannot be both an initial and expansion strategy."
                    )
                exp_strats.append(PointPlacementFactory())
            elif strat == "row":
                exp_strats.append(RowInsertionFactory())
            elif strat == "col":
                exp_strats.append(ColInsertionFactory())
            else:
                raise ValueError(f"Unknown expansion strategy: {strat}")

        if cell_insertion == "expansion":
            exp_strats.append(CellInsertionFactory())
            pack = pack.add_expansion(exp_strats, name_ext=exp_name)
        elif cell_insertion == "initial":
            pack = pack.add_expansion(exp_strats, name_ext=exp_name)
            pack = pack.add_initial(CellInsertionFactory(), "initial_cell_insertion")
        elif cell_insertion == "none":
            pack = pack.add_expansion(exp_strats, name_ext=exp_name)
        else:
            raise ValueError(
                "cell_insertion must be 'initial', 'expansion', or 'none'."
            )

        if point_placement_initial:
            pack = pack.add_initial(PointPlacementFactory(), "initial_point_placement")

        if fusion:
            pack = pack.add_fusion(fusion_type=fusion, apply_first=False)

        if shuffle_factors:
            pack = pack.remove_strategy(FactorStrategy())
            pack = pack.add_initial(ShuffleFactorStrategy(), "with_shuffle_factors")

        if verify:
            if (verify is True or "subclass" in verify) and root is None:
                raise ValueError("root Tiling must be provided for verification.")
            pack = pack.add_verification_strats(strats_to_add=verify, root=root)

        return pack.change_name(pack.name[1:])

    @classmethod
    def basics(cls):
        """Minimum strategies for a pack.
        NOTE: Has no expansion strategies!! Add them in with make_pack."""
        return TileScopePack(
            inferral_strats=[
                RemoveEmptyRowsAndColumnsStrategy(),
                LessThanRowColSeparationStrategy(),
            ],  # Iterable[Strategy]
            initial_strats=[
                FactorStrategy(),
                LessThanOrEqualRowColSeparationFactory(),
                ObstructionTransitivityStrategy(),
            ],  # Iterable[Strategy]
            expansion_strats=[],  # Iterable[Iterable[Strategy]]
            ver_strats=[
                AtomStrategy(),
            ],  # Iterable[Strategy]
            name="",
            symmetries=[],
            iterative=False,
        )

    def add_verification_strats(
        self,
        strats_to_add: list[str] | bool = True,
        root: Tiling | None = None,
    ):
        """Return a new pack with the given verification strategies added.
        strats_to_add: The verification strategies to add, a list of strings. Don't change
        to add all verification strategies.
            - "vert_ins_enc": VerticalInsertionEncodableVerificationStrategy
            - "hori_ins_enc": HorizontalInsertionEncodableVerificationStrategy
            - "subclass": SubclassVerificationStrategy
        root: The root tiling, must be added for subclass verification."""
        if not strats_to_add:
            raise ValueError("At least one verification strategy must be provided.")
        if strats_to_add is True:
            strats_to_add = ["vert_ins_enc", "hori_ins_enc", "subclass"]

        if tuple(sorted(strats_to_add)) == tuple(
            sorted(["vert_ins_enc", "hori_ins_enc", "subclass"])
        ):
            name_ext = "with_verification"
        elif tuple(sorted(strats_to_add)) == tuple(
            sorted(["vert_ins_enc", "hori_ins_enc"])
        ):
            name_ext = "with_insertion_encoding_verification"
        else:
            name_ext = "with_" + "_".join(strats_to_add) + "_verification"

        ver_strats: list[VerificationStrategy] = []
        for strat in strats_to_add:
            if strat == "vert_ins_enc":
                ver_strats.append(VerticalInsertionEncodableVerificationStrategy())
            elif strat == "hori_ins_enc":
                ver_strats.append(HorizontalInsertionEncodableVerificationStrategy())
            elif strat == "subclass":
                if root is None:
                    raise ValueError(
                        "root tiling must be provided for subclass verification."
                    )
                ver_strats.append(SubclassVerificationStrategy(root))
            else:
                raise ValueError(f"Unknown verification strategy: {strat}")

        pack = self.add_verification(ver_strats[0], name_ext=name_ext)
        for other_ver_strats in ver_strats[1:]:
            pack = pack.add_verification(other_ver_strats)
        return pack

    def add_fusion(
        self,
        fusion_type: int = 3,
        apply_first: bool = False,
    ) -> "TileScopePack":
        """
        Create a new pack by adding fusion to the current pack.

        If fusion_type is 1, use standard fusion.
        If fusion_type is 2, use point row fusion.
        If fusion_type is 3, use both.
        """
        if fusion_type == 1:
            return self.add_initial(
                FusionFactory(), "with_fusion", apply_first=apply_first
            )
        if fusion_type == 2:
            return self.add_initial(
                FusionPointRowFactory(),
                "with_point_row_fusion",
                apply_first=apply_first,
            )
        if fusion_type == 3:
            pack = self.add_initial(
                FusionFactory(), "with_fusion", apply_first=apply_first
            )
            pack = pack.add_initial(
                FusionPointRowFactory(),
                "and_point_row_fusion",
                apply_first=apply_first,
            )
            return pack
        raise ValueError("fusion_type must be 1, 2, or 3.")

    def change_name(self, new_name: str) -> "TileScopePack":
        """Return a new pack with the given name."""
        return TileScopePack(
            initial_strats=self.initial_strats,
            inferral_strats=self.inferral_strats,
            expansion_strats=self.expansion_strats,
            ver_strats=self.ver_strats,
            name=new_name,
            symmetries=self.symmetries,
            iterative=self.iterative,
        )

    @classmethod
    def all_packs(cls, root: Tiling):
        """Reuturns a list of all strategy packs with verification."""
        return [
            cls.point_placement_with_verification(root),
            cls.row_and_col_placement_with_verification(root),
            cls.point_and_row_and_col_placement_with_verification(root),
            cls.row_placement_with_verification(root),
            cls.col_placement_with_verification(root),
            cls.initial_point_placement_with_verification(root),
            cls.point_placement_initial_cell_insertion_with_insertion_encoding_verification(),
            cls.point_placement_with_shuffle_factors_with_verification(root),
            cls.point_placement_initial_cell_insertion_with_shuffle_factors_with_verification(
                root
            ),
            cls.point_and_row_and_col_placement_initial_cell_insertion_with_verification(
                root
            ),
        ]

    @classmethod
    def all_packs_atom_ver(cls):
        """Reuturns a list of all strategy packs with only atom verification."""
        return [
            cls.point_placement(),
            cls.row_and_col_placement(),
            cls.make_pack(
                expansions=["point", "row", "col"],
                cell_insertion="expansion",
                verify=False,
            ),
            cls.row_placement(),
            cls.col_placement(),
        ]

    @classmethod
    def point_placement(cls):
        """Point placement strategy pack."""
        return cls.make_pack(expansions=["point"], cell_insertion="expansion")

    @classmethod
    def point_placement_with_verification(cls, root: Tiling):
        """Point placement strategy pack with verification strategies."""
        return cls.make_pack(
            expansions=["point"],
            cell_insertion="expansion",
            root=root,
            verify=True,
        )

    @classmethod
    def initial_point_placement_with_verification(cls, root: Tiling):
        """Point placements strategy pack, place points initially,
        with full verification."""
        return cls.make_pack(
            expansions=[],
            verify=True,
            root=root,
            point_placement_initial=True,
            cell_insertion="expansion",
        )

    @classmethod
    def point_placement_initial_cell_insertion_with_insertion_encoding_verification(
        cls,
    ):
        """Point placements strategy pack with cell insertion
        as an initial strategy."""
        return cls.make_pack(
            expansions=["point"],
            cell_insertion="initial",
            verify=["vert_ins_enc", "hori_ins_enc"],
        )

    @classmethod
    def point_placement_with_shuffle_factors_with_verification(cls, root: Tiling):
        """Point placements with shuffle factors strategy pack and verification."""
        return cls.make_pack(
            expansions=["point"],
            cell_insertion="expansion",
            shuffle_factors=True,
            verify=True,
            root=root,
        )

    @classmethod
    def point_placement_initial_cell_insertion_with_shuffle_factors_with_verification(
        cls, root: Tiling
    ):
        """Point placements with shuffle strategy pack with cell
        insertion as an initial strategy with verification."""
        return cls.make_pack(
            expansions=["point"],
            cell_insertion="initial",
            shuffle_factors=True,
            verify=True,
            root=root,
        )

    @classmethod
    def point_and_row_and_col_placement_with_verification(cls, root: Tiling):
        """Point, row and column placements strategy pack with verification.."""
        return cls.make_pack(
            expansions=["point", "row", "col"],
            cell_insertion="expansion",
            verify=True,
            root=root,
        )

    @classmethod
    def point_and_row_and_col_placement_initial_cell_insertion_with_verification(
        cls, root: Tiling
    ):
        """Point, row and column placements strategy pack with cell insertion
        as an initial strategy and verification."""
        return cls.make_pack(
            expansions=["point", "row", "col"],
            cell_insertion="initial",
            verify=True,
            root=root,
        )

    @classmethod
    def row_and_col_placement(cls):
        """Row and column placement strategy pack."""
        return cls.make_pack(expansions=["row", "col"])

    @classmethod
    def row_placement(cls):
        """Row placement strategy pack."""
        return cls.make_pack(expansions=["row"])

    @classmethod
    def col_placement(cls):
        """Column placement strategy pack."""
        return cls.make_pack(expansions=["col"])

    @classmethod
    def row_and_col_placement_with_verification(cls, root: Tiling):
        """Row and column placement strategy pack with verification."""
        return cls.make_pack(expansions=["row", "col"], verify=True, root=root)

    @classmethod
    def row_placement_with_verification(cls, root: Tiling):
        """Row placement strategy pack with verification."""
        return cls.make_pack(expansions=["row"], verify=True, root=root)

    @classmethod
    def col_placement_with_verification(cls, root: Tiling):
        """Column placement strategy pack with verification."""
        return cls.make_pack(expansions=["col"], verify=True, root=root)

    # Other packs

    @classmethod
    def vertical_insertion_encoding(cls):
        """Vertical insertion encoding strategy pack."""
        return TileScopePack(
            initial_strats=[
                FactorStrategy(),
                VerticalInsertionEncodingRequirementInsertionFactory(),
            ],
            inferral_strats=[RemoveEmptyRowsAndColumnsStrategy()],
            expansion_strats=[[VerticalInsertionEncodingPlacementFactory()]],
            ver_strats=[AtomStrategy()],
            name="vertical_insertion_encoding",
            symmetries=[],
            iterative=False,
        )

    @classmethod
    def horizontal_insertion_encoding(cls):
        """Horizontal insertion encoding strategy pack."""
        return TileScopePack(
            initial_strats=[
                FactorStrategy(),
                HorizontalInsertionEncodingRequirementInsertionFactory(),
            ],
            inferral_strats=[RemoveEmptyRowsAndColumnsStrategy()],
            expansion_strats=[[HorizontalInsertionEncodingPlacementFactory()]],
            ver_strats=[AtomStrategy()],
            name="horizontal_insertion_encoding",
            symmetries=[],
            iterative=False,
        )

    @classmethod
    def cell_insertion(cls, length: int):
        """Cell insertion strategy pack."""
        return TileScopePack(
            inferral_strats=[],
            initial_strats=[],
            expansion_strats=[
                [
                    CellInsertionFactory(maxreqlen=length),
                ]
            ],
            ver_strats=[
                AtomStrategy(),
                VerticalInsertionEncodableVerificationStrategy(),
                HorizontalInsertionEncodableVerificationStrategy(),
                SubclassVerificationStrategy(),
            ],
            name=f"cell_insertion_patterns_up_to_length_{length}",
            symmetries=[],
            iterative=False,
        )
