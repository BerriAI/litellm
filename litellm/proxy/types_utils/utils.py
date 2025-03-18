import importlib
import os
from typing import Any, Callable, Literal, Optional, get_type_hints


def get_instance_fn(value: str, config_file_path: Optional[str] = None) -> Any:
    module_name = value
    instance_name = None
    try:
        # Split the path by dots to separate module from instance
        parts = value.split(".")

        # The module path is all but the last part, and the instance_name is the last part
        module_name = ".".join(parts[:-1])
        instance_name = parts[-1]

        # If config_file_path is provided, use it to determine the module spec and load the module
        if config_file_path is not None:
            directory = os.path.dirname(config_file_path)
            module_file_path = os.path.join(directory, *module_name.split("."))
            module_file_path += ".py"

            spec = importlib.util.spec_from_file_location(module_name, module_file_path)  # type: ignore
            if spec is None:
                raise ImportError(
                    f"Could not find a module specification for {module_file_path}"
                )
            module = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(module)  # type: ignore
        else:
            # Dynamically import the module
            module = importlib.import_module(module_name)

        # Get the instance from the module
        instance = getattr(module, instance_name)

        return instance
    except ImportError as e:
        # Re-raise the exception with a user-friendly message
        if instance_name and module_name:
            raise ImportError(
                f"Could not import {instance_name} from {module_name}"
            ) from e
        else:
            raise e
    except Exception as e:
        raise e


def validate_custom_validate_return_type(
    fn: Optional[Callable[..., Any]]
) -> Optional[Callable[..., Literal[True]]]:
    if fn is None:
        return None

    hints = get_type_hints(fn)
    return_type = hints.get("return")

    if return_type != Literal[True]:
        raise TypeError(
            f"Custom validator must be annotated to return Literal[True], got {return_type}"
        )

    return fn
