"""Pytest plugin to swallow coverage options when pytest-cov is unavailable."""


def pytest_addoption(parser) -> None:
    parser.addoption("--cov", action="append", default=[], help="No-op placeholder for pytest-cov")
    parser.addoption(
        "--cov-report",
        action="append",
        default=[],
        help="No-op placeholder for pytest-cov",
    )
