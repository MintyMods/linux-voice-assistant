"""Test bootstrap: stub native deps that the test surface doesn't actually exercise.

These tests cover the generation-counter contract in satellite.py, which never
touches the real mpv/soundcard/webrtc backends. Stubbing them at sys.modules
lets the tests run on dev machines without libmpv installed.
"""

import sys
import types


def _install_stub(name: str) -> None:
    if name in sys.modules:
        return
    sys.modules[name] = types.ModuleType(name)


# libmpv + netifaces are loaded transitively on import of
# linux_voice_assistant.{entity,util} → satellite. Tests never call into them.
_install_stub("mpv")
_install_stub("netifaces")
