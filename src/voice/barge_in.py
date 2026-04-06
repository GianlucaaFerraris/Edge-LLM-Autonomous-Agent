"""
barge_in.py — Detector de interrupciones por voz (barge-in) v2.

Permite al usuario interrumpir al asistente mientras habla.

DISEÑO v2 — "Always-on VAD con umbral alto":
  El VAD corre continuamente mientras el TTS reproduce. Para discriminar
  el echo del speaker (que el mic captura) de la voz real del usuario,
  usamos un umbral de energía significativamente más alto que el del STT.

  Justificación física:
    - Echo speaker→mic (laptop): ~2000-7000 RMS (atenuado por distancia
      speaker-mic, ángulo, y response del mic)
    - Voz directa usuario→mic (~20-40cm): ~12000-30000 RMS
    - Ratio típico: 3x-5x entre voz directa y echo

  El umbral se setea en ~12000 RMS — por encima del echo típico pero
  por debajo de la voz directa. Esto funciona en la mayoría de laptops
  y setups de SBC con mic USB + speaker separado.

  Si el setup tiene echo muy fuerte (speaker potente cerca del mic),
  se puede subir con env AGENTY_BARGE_THRESHOLD.

PROTOCOLO:
  1. monitor.start()           → arranca thread de captura de mic
  2. [speak_stream reproduce]  → monitor escucha continuamente
  3. monitor.interrupted       → True si detectó interrupción
  4. monitor.stop()            → cierra stream, limpia estado
"""

import os
import threading
import queue
import time
import numpy as np
from typing import Optional


# ── Configuración ─────────────────────────────────────────────────────────────

SAMPLE_RATE  = 16000
CHANNELS     = 1
CHUNK_SIZE   = 512     # ~32ms por chunk a 16kHz
DTYPE        = "int16"

# Umbral de energía para barge-in.
# Debe ser MÁS ALTO que el echo del speaker para evitar false positives.
# Override: export AGENTY_BARGE_THRESHOLD=15000
DEFAULT_BARGE_THRESHOLD = 12000

# Chunks consecutivos con voz para confirmar interrupción.
# 5 chunks × 32ms = ~160ms de voz sostenida.
CONFIRM_CHUNKS = 5

# Chunks de gracia: si hay 1-2 chunks de silencio entre chunks de voz,
# no resetear el contador (pausa natural entre palabras).
GRACE_CHUNKS = 2


def _get_threshold() -> int:
    """Lee threshold de env var o usa default."""
    env = os.environ.get("AGENTY_BARGE_THRESHOLD")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return DEFAULT_BARGE_THRESHOLD


# ── Monitor ───────────────────────────────────────────────────────────────────

class BargeInMonitor:
    """
    Monitor de barge-in: escucha el mic continuamente durante el TTS
    y detecta cuando el usuario habla con energía suficiente para
    ser voz directa (no echo del speaker).

    Thread-safe: el thread de captura escribe, el thread principal lee.
    Comunicación via threading.Event (lock-free).
    """

    def __init__(self, device_idx: Optional[int] = None,
                 energy_threshold: int = None):
        self._device_idx = device_idx
        self._energy_threshold = energy_threshold or _get_threshold()

        # Estado
        self._interrupted = threading.Event()
        self._running = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        # Debug: último y pico RMS medido (para calibración)
        self._last_rms: float = 0.0
        self._peak_rms: float = 0.0

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    def start(self) -> bool:
        """
        Inicia captura de audio para monitoreo.
        Retorna True si el mic se pudo abrir.
        """
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._interrupted.clear()
            self._peak_rms = 0.0
            return True

        self._interrupted.clear()
        self._running.set()
        self._peak_rms = 0.0

        try:
            import sounddevice as sd
            sd.check_input_settings(
                device=self._device_idx,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
            )
        except Exception as e:
            print(f"  [BARGE-IN] ❌ No se puede abrir mic para barge-in: {e}")
            return False

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="barge-in-vad",
            daemon=True,
        )
        self._capture_thread.start()
        print(f"  [BARGE-IN] ✅ Monitor activo (threshold={self._energy_threshold})")
        return True

    def stop(self) -> None:
        """Detiene captura. Imprime peak RMS para calibración."""
        self._running.clear()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        if self._peak_rms > 0:
            print(f"  [BARGE-IN] Sesión finalizada — peak RMS: {self._peak_rms:.0f} "
                  f"(threshold: {self._energy_threshold})")

    def reset(self) -> None:
        """Limpia flag de interrupción para reutilizar."""
        self._interrupted.clear()
        self._peak_rms = 0.0

    # ── Capture thread ────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Loop de captura + VAD en thread daemon.
        Corre durante toda la duración del speak_stream.
        """
        import sounddevice as sd

        audio_q: queue.Queue[bytes] = queue.Queue(maxsize=100)

        def _callback(indata: bytes, frames: int, time_info, status) -> None:
            try:
                audio_q.put_nowait(bytes(indata))
            except queue.Full:
                pass

        try:
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK_SIZE,
                device=self._device_idx,
                callback=_callback,
            )
        except Exception as e:
            print(f"  [BARGE-IN] ❌ Error abriendo stream de mic: {e}")
            print(f"  [BARGE-IN]    Device idx: {self._device_idx}")
            return

        consecutive_voice = 0
        silence_grace = 0

        with stream:
            while self._running.is_set():
                try:
                    raw = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._interrupted.is_set():
                    continue

                # VAD de energía RMS
                chunk_np = np.frombuffer(raw, dtype=np.int16)
                rms = float(np.sqrt(np.mean(chunk_np.astype(np.float32) ** 2)))
                self._last_rms = rms
                if rms > self._peak_rms:
                    self._peak_rms = rms

                if rms > self._energy_threshold:
                    consecutive_voice += 1
                    silence_grace = 0
                    if consecutive_voice >= CONFIRM_CHUNKS:
                        self._interrupted.set()
                        print(f"\n  [BARGE-IN] 🛑 Interrupción detectada "
                              f"(RMS={rms:.0f}, threshold={self._energy_threshold})")
                else:
                    if consecutive_voice > 0:
                        silence_grace += 1
                        if silence_grace > GRACE_CHUNKS:
                            consecutive_voice = 0
                            silence_grace = 0


# ── Singleton ─────────────────────────────────────────────────────────────────

_monitor: Optional[BargeInMonitor] = None


def get_barge_in_monitor(device_idx: Optional[int] = None) -> BargeInMonitor:
    global _monitor
    if _monitor is None:
        _monitor = BargeInMonitor(device_idx=device_idx)
    return _monitor