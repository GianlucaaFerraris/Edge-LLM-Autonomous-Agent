"""
language_tool_server.py — Lifecycle del servidor LanguageTool.

Levanta el servidor LT como subproceso al entrar al tutor de inglés
y lo apaga limpiamente al salir.

Uso:
    from src.english.language_tool_server import LTServer

    with LTServer() as lt:
        if lt.running:
            # LT disponible
            errors = check_errors(text)

O manualmente:
    lt = LTServer()
    lt.start()
    ...
    lt.stop()

Requisitos:
    - Java instalado (java -version debe funcionar)
    - JAR en ~/languagetool/LanguageTool-*/languagetool-server.jar
      (o configurar LT_JAR_PATH en variables de entorno)
"""

import os
import subprocess
import time
import requests
import glob
import signal

LT_PORT    = 8081
LT_URL     = f"http://localhost:{LT_PORT}/v2/languages"
LT_TIMEOUT = 30  # segundos máximos para esperar que arranque


def _find_jar() -> str | None:
    env_path = os.environ.get("LT_JAR_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    patterns = [
        os.path.expanduser("~/languagetool/LanguageTool-*/languagetool-server.jar"),
        os.path.expanduser("~/LanguageTool-*/languagetool-server.jar"),
        "/opt/languagetool/languagetool-server.jar",
        "/usr/local/lib/languagetool/languagetool-server.jar",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]
    return None


class LTServer:
    """
    Maneja el ciclo de vida del servidor LanguageTool.
    Soporta context manager (with statement).
    """

    def __init__(self, jar_path: str = None):
        self.jar_path = jar_path or _find_jar()
        self._process: subprocess.Popen | None = None
        self.running  = False
        self._already_running = False

    def _is_up(self) -> bool:
        try:
            resp = requests.get(LT_URL, timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def start(self) -> bool:
        if self._is_up():
            if self._process is None:
                print("  [LT] Servidor ya estaba corriendo — usando instancia existente.")
                self._already_running = True
            else:
                print("  [LT] Proceso propio aún activo — reusando.")
            self.running = True
            return True

        if not self.jar_path:
            print("  [LT] ⚠ JAR no encontrado. Opciones:")
            print("       1. Descargar desde https://languagetool.org/download/")
            print("       2. Extraer en ~/languagetool/")
            print("       3. O setear: export LT_JAR_PATH=/ruta/al/languagetool-server.jar")
            print("  [LT] El tutor funcionará SIN detección de errores.")
            self.running = False
            return False

        jar_dir = os.path.dirname(self.jar_path)
        print(f"  [LT] Arrancando servidor desde: {self.jar_path}")

        try:
            self._process = subprocess.Popen(
                [
                    "java", "-cp", self.jar_path,
                    "org.languagetool.server.HTTPServer",
                    "--port", str(LT_PORT),
                    "--allow-origin", "*",
                    "--public",
                ],
                cwd=jar_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            print("  [LT] ⚠ Java no encontrado. Instalalo con: sudo apt install default-jre")
            self.running = False
            return False
        except Exception as e:
            print(f"  [LT] ⚠ Error al arrancar: {e}")
            self.running = False
            return False

        print(f"  [LT] Esperando que el servidor levante (máx {LT_TIMEOUT}s)...", end="", flush=True)
        deadline = time.time() + LT_TIMEOUT
        while time.time() < deadline:
            if self._is_up():
                print(" ✅")
                self.running = True
                return True
            time.sleep(1)
            print(".", end="", flush=True)

        print(" ❌ Timeout")
        self.stop()
        self.running = False
        return False

    @staticmethod
    def _kill_port(port: int) -> bool:
        """Mata cualquier proceso escuchando en el puerto dado. Retorna True si mató algo."""
        try:
            # fuser es más liviano que lsof en sistemas embebidos
            result = subprocess.run(
                ["fuser", "-k", "-TERM", f"{port}/tcp"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                time.sleep(1)  # grace period para SIGTERM
                # Si aún responde, SIGKILL
                subprocess.run(
                    ["fuser", "-k", "-KILL", f"{port}/tcp"],
                    capture_output=True, timeout=3
                )
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    def stop(self) -> None:
        if self._process is not None:
            # Tenemos el handle — matar por grupo de procesos
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                    self._process.wait(timeout=3)
                except Exception:
                    pass
            except (ProcessLookupError, OSError):
                pass
            except Exception as e:
                print(f"  [LT] Error al apagar: {e}")
        elif self._already_running:
            # No tenemos el handle pero sabemos que está en el puerto — matar por puerto
            self._kill_port(LT_PORT)

        self._process = None
        self._already_running = False
        self.running = False

        # Confirmar muerte del puerto (máx 5s)
        deadline = time.time() + 5
        while time.time() < deadline:
            if not self._is_up():
                print("  [LT] Servidor apagado.")
                break
            time.sleep(0.3)
        else:
            print("  [LT] ⚠ El servidor sigue respondiendo tras el shutdown.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# ── Singleton global ──────────────────────────────────────────────────────────
_lt_server: LTServer | None = None


def get_server() -> LTServer:
    global _lt_server
    if _lt_server is None:
        _lt_server = LTServer()
    return _lt_server


def ensure_running() -> bool:
    return get_server().start()


def ensure_stopped() -> None:
    if _lt_server is not None:
        _lt_server.stop()