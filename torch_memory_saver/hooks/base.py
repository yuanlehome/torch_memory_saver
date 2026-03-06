from abc import ABC
from typing import Literal

HookMode = Literal["preload"]


class HookUtilBase(ABC):
    @staticmethod
    def create(hook_mode: HookMode) -> "HookUtilBase":
        from torch_memory_saver.hooks.mode_preload import HookUtilModePreload
        assert hook_mode == "preload", f"Only hook_mode=preload is supported, got {hook_mode!r}"
        return HookUtilModePreload()

    def get_path_binary(self):
        raise NotImplementedError
