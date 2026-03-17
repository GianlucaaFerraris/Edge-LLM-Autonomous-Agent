"""
context_manager.py — Estado compartido entre modos del orquestador.

Mantiene:
  - modo activo actual
  - historial de cada modo (para reanudar donde se dejó)
  - instancias de sesión activas
"""

from dataclasses import dataclass, field
from typing import Optional, Any


MODES = ("idle", "english", "engineering", "agent", "web_search")


@dataclass
class ModeContext:
    """Estado de una sesión de un modo específico."""
    mode:     str
    history:  list[dict] = field(default_factory=list)
    session:  Any        = None   # instancia de TutorSession / AgentSession / etc.
    topic:    str        = ""     # para english tutor
    metadata: dict       = field(default_factory=dict)


class ContextManager:
    """Gestiona el estado global entre modos."""

    def __init__(self):
        self.active_mode:    str                  = "idle"
        self.previous_mode:  Optional[str]        = None
        self._contexts:      dict[str, ModeContext] = {}

    def get(self, mode: str) -> ModeContext:
        if mode not in self._contexts:
            self._contexts[mode] = ModeContext(mode=mode)
        return self._contexts[mode]

    def set_active(self, mode: str) -> None:
        if mode != self.active_mode:
            self.previous_mode = self.active_mode
            self.active_mode   = mode

    def return_to_previous(self) -> Optional[str]:
        """Vuelve al modo anterior. Retorna el nombre del modo."""
        if self.previous_mode and self.previous_mode != "idle":
            mode = self.previous_mode
            self.set_active(mode)
            return mode
        return None

    def clear(self, mode: str) -> None:
        """Limpia el contexto de un modo (nueva sesión)."""
        if mode in self._contexts:
            del self._contexts[mode]

    @property
    def has_active_session(self) -> bool:
        return self.active_mode not in ("idle",)

    @property
    def active_context(self) -> ModeContext:
        return self.get(self.active_mode)