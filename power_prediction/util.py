from pathlib import Path


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    try:
        project_root = Path(__file__).parent.parent
    except NameError:
        project_root = Path().resolve()
    return project_root