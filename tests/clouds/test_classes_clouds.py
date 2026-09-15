from clouds import TrackedTileScopePack, TrackedSearcher
from cayley_permutations import Av


def test_some_classes_clouds1():
    """Check that we can still find specs for these classes."""
    basis = "012_101_001_010"
    pack = TrackedTileScopePack.standard_fusion_pack(expansion_methods=["point"])
    searcher = TrackedSearcher(basis, pack, debug=False, max_cvs=1)
    spec = searcher.auto_search(status_update=5)

    spec = spec.expand_verified()
    spec.sanity_check(4)
    spec_counts = [spec.count_objects_of_size(i) for i in range(10)]
    root_counts = Av(basis).counter(9)
    assert spec_counts == root_counts


def test_some_classes_clouds2():
    """Check that we can still find specs for these classes."""
    basis = "012_101_010"
    pack = TrackedTileScopePack.standard_fusion_pack(expansion_methods=["point"])
    searcher = TrackedSearcher(basis, pack, debug=False, max_cvs=1)
    spec = searcher.auto_search(status_update=5)

    spec = spec.expand_verified()
    spec.sanity_check(4)
    spec_counts = [spec.count_objects_of_size(i) for i in range(10)]
    root_counts = Av(basis).counter(9)
    assert spec_counts == root_counts


def test_some_classes_clouds3():
    """Check that we can still find specs for these classes."""
    basis = "210_101_011"
    pack = TrackedTileScopePack.standard_fusion_pack(expansion_methods=["point"])
    searcher = TrackedSearcher(basis, pack, debug=False, max_cvs=1)
    spec = searcher.auto_search(status_update=5)

    spec = spec.expand_verified()
    spec.sanity_check(4)
    spec_counts = [spec.count_objects_of_size(i) for i in range(10)]
    root_counts = Av(basis).counter(9)
    assert spec_counts == root_counts


def test_some_classes_clouds4():
    """Check that we can still find specs for these classes."""
    basis = "101_010_011"
    pack = TrackedTileScopePack.standard_fusion_pack(expansion_methods=["point"])
    searcher = TrackedSearcher(basis, pack, debug=False, max_cvs=1)
    spec = searcher.auto_search(status_update=5)

    spec = spec.expand_verified()
    spec.sanity_check(4)
    spec_counts = [spec.count_objects_of_size(i) for i in range(10)]
    root_counts = Av(basis).counter(9)
    assert spec_counts == root_counts


def test_some_classes_clouds5():
    """Check that we can still find specs for these classes."""
    basis = "100_110_010_011"
    pack = TrackedTileScopePack.standard_fusion_pack(expansion_methods=["point"])
    searcher = TrackedSearcher(basis, pack, debug=False, max_cvs=1)
    spec = searcher.auto_search(status_update=5)

    spec = spec.expand_verified()
    spec.sanity_check(4)
    spec_counts = [spec.count_objects_of_size(i) for i in range(10)]
    root_counts = Av(basis).counter(9)
    assert spec_counts == root_counts


def test_some_classes_clouds6():
    """Check that we can still find specs for these classes."""
    basis = "101_010"
    pack = TrackedTileScopePack.standard_fusion_pack(expansion_methods=["point"])
    searcher = TrackedSearcher(basis, pack, debug=False, max_cvs=1)
    spec = searcher.auto_search(status_update=5)

    spec = spec.expand_verified()
    spec.sanity_check(4)
    spec_counts = [spec.count_objects_of_size(i) for i in range(10)]
    root_counts = Av(basis).counter(9)
    assert spec_counts == root_counts


def test_some_classes_clouds7():
    """Check that we can still find specs for these classes."""
    basis = "101_110_010_001_011"
    pack = TrackedTileScopePack.standard_fusion_pack(expansion_methods=["point"])
    searcher = TrackedSearcher(basis, pack, debug=False, max_cvs=1)
    spec = searcher.auto_search(status_update=5)

    spec = spec.expand_verified()
    spec.sanity_check(4)
    spec_counts = [spec.count_objects_of_size(i) for i in range(10)]
    root_counts = Av(basis).counter(9)
    assert spec_counts == root_counts


def test_some_classes_clouds8():
    """Check that we can still find specs for these classes."""
    basis = "100_110_001_011"
    pack = TrackedTileScopePack.standard_fusion_pack(expansion_methods=["point"])
    searcher = TrackedSearcher(basis, pack, debug=False, max_cvs=1)
    spec = searcher.auto_search(status_update=5)

    spec = spec.expand_verified()
    spec.sanity_check(4)
    spec_counts = [spec.count_objects_of_size(i) for i in range(10)]
    root_counts = Av(basis).counter(9)
    assert spec_counts == root_counts


def test_some_classes_clouds9():
    """Check that we can still find specs for these classes."""
    basis = "000_010_100_101_210"
    pack = TrackedTileScopePack.standard_fusion_pack(expansion_methods=["point"])
    searcher = TrackedSearcher(basis, pack, debug=False, max_cvs=1)
    spec = searcher.auto_search(status_update=5)

    spec = spec.expand_verified()
    spec.sanity_check(4)
    spec_counts = [spec.count_objects_of_size(i) for i in range(10)]
    root_counts = Av(basis).counter(9)
    assert spec_counts == root_counts


def test_some_classes_clouds19():
    """Check that we can still find specs for these classes."""
    basis = "012"
    pack = TrackedTileScopePack.standard_fusion_pack(expansion_methods=["point"])
    searcher = TrackedSearcher(basis, pack, debug=False, max_cvs=1)
    spec = searcher.auto_search(status_update=5)

    spec = spec.expand_verified()
    spec.sanity_check(4)
    spec_counts = [spec.count_objects_of_size(i) for i in range(10)]
    root_counts = Av(basis).counter(9)
    assert spec_counts == root_counts
