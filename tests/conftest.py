"""Pytest configuration — handles test isolation between mocked and real-torch suites.

test_cli_train.py mocks torch only inside specific tests (not at module level),
so frame_alignment tests can run in the same session as long as they execute first.
"""


def pytest_collection_modifyitems(config, items):
    """Run real-torch tests before cli_train tests that may mock torch.

    frame_alignment and parallel_extraction both require real PyTorch/torchaudio.
    cli_train mocks torch inside specific test methods — if those run first,
    the mocked modules corrupt sys.modules for subsequent real imports.
    """
    def _sort_key(item):
        # Group 0: tests that need real torch (run FIRST)
        if "frame_alignment" in item.nodeid or "parallel_extraction" in item.nodeid:
            return 0
        # Group 1: everything else (cli_train, etc.)
        return 1

    items.sort(key=_sort_key)
