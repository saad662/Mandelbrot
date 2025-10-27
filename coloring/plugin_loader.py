import importlib
import traceback
from coloring.base import ColoringInterface


def load_coloring_plugin(path: str) -> ColoringInterface:
    try:
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)

        cls = getattr(module, class_name)

        if not issubclass(cls, ColoringInterface):
            raise TypeError(f"{path} is not a subclass of {ColoringInterface}")

        return cls()

    except Exception as e:
        print("[Plugin Loader Error]")
        print("Exception:", e)
        traceback.print_exc()
        raise  # Re-raise the exception so you don't silently fail
