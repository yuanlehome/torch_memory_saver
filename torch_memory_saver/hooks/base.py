from abc import ABC


class HookUtilBase(ABC):
    @staticmethod
    def create() -> "HookUtilBase":
        from torch_memory_saver.hooks.mode_preload import HookUtilModePreload
        return HookUtilModePreload()

    def get_path_binary(self):
        raise NotImplementedError
