from importlib.resources import files


def load(package: str, name: str) -> str:
    return files(package).joinpath(f"{name}.sql").read_text(encoding="utf-8")
