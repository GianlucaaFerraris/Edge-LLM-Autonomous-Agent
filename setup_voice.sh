#!/bin/bash
# setup_voice.sh — Instalación de dependencias de voz para Agenty.
#
# Ejecutar: bash setup_voice.sh
#
# Plataformas soportadas:
#   - x86_64 Linux (para desarrollo)
#   - aarch64 Linux (Rock 5B, Jetson Orin Nano)
#
# Componentes:
#   1. Moonshine Voice v2 (STT) — modelos en/es
#   2. Piper TTS + voces seleccionadas
#   3. Audio tools (aplay, sox, sounddevice)
#   4. Dependencias de Python

set -e

echo "═══════════════════════════════════════════════════════"
echo "  Agenty Voice Setup"
echo "═══════════════════════════════════════════════════════"
echo ""

ARCH=$(uname -m)
echo "  Arquitectura: $ARCH"
echo ""

# ── 1. Dependencias del sistema ───────────────────────────────────────────────

echo "▶ Instalando dependencias del sistema..."

if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3-pip \
        python3-dev \
        libasound2-dev \
        portaudio19-dev \
        alsa-utils \
        sox \
        libsox-fmt-all \
        ffmpeg
elif command -v dnf &> /dev/null; then
    sudo dnf install -y \
        python3-pip \
        python3-devel \
        alsa-lib-devel \
        portaudio-devel \
        alsa-utils \
        sox \
        ffmpeg
else
    echo "  ⚠ Package manager no reconocido. Instalá manualmente:"
    echo "    - portaudio, alsa-utils, sox, ffmpeg"
fi

echo "  ✅ Dependencias del sistema instaladas."
echo ""

# ── 2. Moonshine Voice (STT) ─────────────────────────────────────────────────

echo "▶ Instalando Moonshine Voice..."

pip install --break-system-packages moonshine-voice 2>/dev/null || \
pip install moonshine-voice

echo "  Descargando modelo de inglés..."
python3 -c "
from moonshine_voice import get_model_for_language
get_model_for_language('en')
print('  ✅ Modelo inglés descargado.')
" 2>/dev/null || echo "  ⚠ No se pudo descargar el modelo de inglés."

echo "  Descargando modelo de español..."
python3 -c "
from moonshine_voice import get_model_for_language
get_model_for_language('es')
print('  ✅ Modelo español descargado.')
" 2>/dev/null || echo "  ⚠ No se pudo descargar el modelo de español."

echo ""

# ── 3. Piper TTS ─────────────────────────────────────────────────────────────

echo "▶ Instalando Piper TTS..."

pip install --break-system-packages piper-tts 2>/dev/null || \
pip install piper-tts

# Dependencias para reproducción de audio
pip install --break-system-packages sounddevice numpy 2>/dev/null || \
pip install sounddevice numpy

echo "  ✅ Piper TTS instalado."
echo ""

# ── 4. Descargar voces de Piper ──────────────────────────────────────────────

echo "▶ Descargando voces de Piper..."

VOICES_DIR="$HOME/.local/share/piper-voices"
mkdir -p "$VOICES_DIR"

HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"

download_voice() {
    local lang_family=$1
    local lang_code=$2
    local name=$3
    local quality=$4
    local desc=$5

    local filename="${lang_code}-${name}-${quality}.onnx"
    local model_url="${HF_BASE}/${lang_family}/${lang_code}/${name}/${quality}/${filename}"
    local config_url="${model_url}.json"

    if [ -f "${VOICES_DIR}/${filename}" ] && [ -f "${VOICES_DIR}/${filename}.json" ]; then
        echo "  ✅ ${desc} (ya descargada)"
        return
    fi

    echo "  ⬇ ${desc}..."
    wget -q -O "${VOICES_DIR}/${filename}" "${model_url}" 2>/dev/null || \
    curl -sL -o "${VOICES_DIR}/${filename}" "${model_url}"

    wget -q -O "${VOICES_DIR}/${filename}.json" "${config_url}" 2>/dev/null || \
    curl -sL -o "${VOICES_DIR}/${filename}.json" "${config_url}"

    if [ -f "${VOICES_DIR}/${filename}" ]; then
        local size=$(du -h "${VOICES_DIR}/${filename}" | cut -f1)
        echo "  ✅ ${desc} (${size})"
    else
        echo "  ❌ Error descargando ${desc}"
    fi
}

# Agente/Orquestador — hombre amigable (es_MX)
download_voice "es" "es_MX" "ald" "medium" \
    "Agente: Hombre amigable (es_MX-ald-medium)"

# Ingeniería español — hombre maduro (es_ES)
download_voice "es" "es_ES" "davefx" "medium" \
    "Ingeniería: Hombre maduro (es_ES-davefx-medium)"

# Ingeniería inglés fallback — hombre (en_US)
download_voice "en" "en_US" "hfc_male" "medium" \
    "Ingeniería EN: Hombre (en_US-hfc_male-medium)"

# Tutor inglés español — mujer argentina (es_AR)
download_voice "es" "es_AR" "daniela" "high" \
    "Tutor EN (español): Mujer (es_AR-daniela-high)"

# Tutor inglés inglés — mujer (en_US)
download_voice "en" "en_US" "amy" "medium" \
    "Tutor EN (inglés): Mujer (en_US-amy-medium)"

echo ""

# ── 5. Verificación ──────────────────────────────────────────────────────────

echo "▶ Verificación..."
echo ""

python3 -c "
import sys

checks = []

# Moonshine
try:
    from moonshine_voice import get_model_for_language
    checks.append(('Moonshine Voice', 'OK'))
except ImportError:
    checks.append(('Moonshine Voice', 'FALTA: pip install moonshine-voice'))

# Piper
try:
    import piper
    checks.append(('Piper TTS (Python)', 'OK'))
except ImportError:
    # Intentar CLI
    import subprocess
    try:
        r = subprocess.run(['piper', '--version'], capture_output=True, timeout=5)
        checks.append(('Piper TTS (CLI)', 'OK'))
    except Exception:
        checks.append(('Piper TTS', 'FALTA: pip install piper-tts'))

# sounddevice
try:
    import sounddevice
    checks.append(('sounddevice', 'OK'))
except ImportError:
    checks.append(('sounddevice', 'FALTA: pip install sounddevice'))

# numpy
try:
    import numpy
    checks.append(('numpy', 'OK'))
except ImportError:
    checks.append(('numpy', 'FALTA: pip install numpy'))

# Voces
import os
voices_dir = os.path.expanduser('~/.local/share/piper-voices')
voice_files = [
    'es_MX-ald-medium.onnx',
    'es_ES-davefx-medium.onnx',
    'en_US-hfc_male-medium.onnx',
    'es_AR-daniela-high.onnx',
    'en_US-amy-medium.onnx',
]
for vf in voice_files:
    path = os.path.join(voices_dir, vf)
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        checks.append((f'Voz {vf}', f'OK ({size:.1f} MB)'))
    else:
        checks.append((f'Voz {vf}', 'FALTA'))

# Mostrar resultados
print('  ┌─────────────────────────────────────────────────────┐')
for name, status in checks:
    icon = '✅' if 'OK' in status else '❌'
    print(f'  │ {icon} {name:35s} {status:15s} │')
print('  └─────────────────────────────────────────────────────┘')

# Exit code
failed = [c for c in checks if 'OK' not in c[1]]
if failed:
    print(f'\n  ⚠ {len(failed)} componente(s) faltantes.')
    sys.exit(1)
else:
    print('\n  ✅ Todo listo. Ejecutá: python main.py')
"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Setup completo."
echo ""
echo "  Modos de ejecución:"
echo "    python main.py                    # voz completa"
echo "    python main.py --keyboard         # teclado + audio"
echo "    python main.py --print-only       # audio + texto"
echo "    python main.py --keyboard --print-only  # texto puro"
echo ""
echo "  Test individual:"
echo "    python stt_engine.py --lang es    # test STT español"
echo "    python stt_engine.py --lang en    # test STT inglés"
echo "    python tts_engine.py              # test TTS todas las voces"
echo "    python voice_io.py --keyboard     # test VoiceIO interactivo"
echo "═══════════════════════════════════════════════════════"