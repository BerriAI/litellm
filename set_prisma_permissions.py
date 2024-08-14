import os
import importlib
from pathlib import Path


# Get the location of the 'prisma' package
package_name = "prisma"
spec = importlib.util.find_spec(package_name)
print("spec = ", spec)  # noqa

if spec and spec.origin:
    print("spec origin= ", spec.origin)  # noqa
    _base_prisma_package_dir = os.path.dirname(spec.origin)
    print("base prisma package dir = ", _base_prisma_package_dir)  # noqa
else:
    raise ImportError(f"Package {package_name} not found.")


def ensure_prisma_has_writable_dirs(path: str | Path) -> None:
    import stat

    for root, dirs, _ in os.walk(path):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            os.makedirs(dir_path, exist_ok=True)
            print("making dir for prisma = ", dir_path)
            os.chmod(dir_path, os.stat(dir_path).st_mode | stat.S_IWRITE | stat.S_IEXEC)

    # make this file writable - prisma/schema.prisma
    file_path = os.path.join(path, "schema.prisma")
    print("making file for prisma = ", file_path)
    # make entire directory writable
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE | stat.S_IEXEC)

    os.chmod(file_path, os.stat(file_path).st_mode | stat.S_IWRITE | stat.S_IEXEC)


# Use the package directory in your method call
ensure_prisma_has_writable_dirs(path=_base_prisma_package_dir)
