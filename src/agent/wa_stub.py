"""
wa_stub.py — Stub de WhatsApp (simulado).

Implementa la misma interfaz que tendrá el módulo real con whatsapp-web.js.
Los contactos se guardan en un JSON local para que el dispatcher
pueda resolver ambigüedades ("Juanma" → "Juan Manuel García").

Para conectar WhatsApp real: reemplazar send() y read() por llamadas
al servidor whatsapp-web.js en localhost.
"""

import json
import os
import datetime

CONTACTS_PATH = os.path.join(os.path.dirname(__file__), "db", "wa_contacts.json")

# Contactos de ejemplo — el usuario puede editarlos directamente
DEFAULT_CONTACTS = [
    {"name": "Mamá",         "phone": "+54911XXXXXXX", "aliases": ["mama", "ma"]},
    {"name": "Papá",         "phone": "+54911XXXXXXX", "aliases": ["papa", "pa"]},
    {"name": "Juan Manuel",  "phone": "+54911XXXXXXX", "aliases": ["juanma", "juan manuel", "juan"]},
    {"name": "Tomi",         "phone": "+54911XXXXXXX", "aliases": ["tomi", "tomás", "tomas"]},
]


def _load_contacts() -> list[dict]:
    os.makedirs(os.path.dirname(CONTACTS_PATH), exist_ok=True)
    if not os.path.exists(CONTACTS_PATH):
        with open(CONTACTS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONTACTS, f, ensure_ascii=False, indent=2)
        return DEFAULT_CONTACTS
    with open(CONTACTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_contact(query: str) -> list[dict]:
    """
    Busca contactos que coincidan con el query (nombre o alias).
    Retorna lista de matches — puede ser 0, 1 o varios.
    """
    q = query.lower().strip()
    contacts = _load_contacts()
    matches = []
    for c in contacts:
        if q in c["name"].lower():
            matches.append(c)
            continue
        if any(q in alias.lower() for alias in c.get("aliases", [])):
            matches.append(c)
    return matches


def list_contacts() -> list[dict]:
    return _load_contacts()


def format_contacts(contacts: list[dict] = None) -> str:
    """Formatea lista de contactos para mostrar al usuario."""
    if contacts is None:
        contacts = _load_contacts()
    if not contacts:
        return "No hay contactos guardados."
    lines = [f"  📱 {c['name']}" for c in contacts]
    return "Contactos disponibles:\n" + "\n".join(lines)


def send(contact_name: str, message: str) -> dict:
    """
    Envía un mensaje (simulado).
    Retorna {success, contact, message, timestamp, simulated}.
    """
    matches = resolve_contact(contact_name)
    if not matches:
        return {
            "success": False,
            "error":   f"No encontré ningún contacto para '{contact_name}'.",
        }
    if len(matches) > 1:
        return {
            "success":   False,
            "ambiguous": True,
            "matches":   [m["name"] for m in matches],
            "error":     f"Encontré {len(matches)} contactos que coinciden.",
        }
    contact = matches[0]
    timestamp = datetime.datetime.now().isoformat()
    # TODO: reemplazar con llamada real a whatsapp-web.js
    print(f"  [WA STUB] → {contact['name']} ({contact['phone']}): {message}")
    return {
        "success":   True,
        "contact":   contact["name"],
        "phone":     contact["phone"],
        "message":   message,
        "timestamp": timestamp,
        "simulated": True,
    }


def read(contact_name: str = None) -> dict:
    """
    Lee mensajes recientes (simulado).
    """
    # TODO: reemplazar con llamada real a whatsapp-web.js
    if contact_name:
        matches = resolve_contact(contact_name)
        name = matches[0]["name"] if matches else contact_name
        return {
            "success":   True,
            "contact":   name,
            "messages":  [{"from": name, "text": "[SIMULADO] Hola!", "time": "10:30"}],
            "simulated": True,
        }
    return {
        "success":   True,
        "messages":  [{"from": "Mamá", "text": "[SIMULADO] Todo bien?", "time": "09:15"}],
        "simulated": True,
    }