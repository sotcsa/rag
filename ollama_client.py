"""
Közös Ollama kliens — 127.0.0.1-re konfigurálva,
hogy WiFi/DNS nélkül is működjön.
"""

import ollama

import config

# Explicit kliens 127.0.0.1 loopback IP-vel — nem kell DNS feloldás
# Timeout: 5 perc (nagy szegmensek chunkolása lassú lehet a 14B modellel)
client = ollama.Client(host=config.OLLAMA_BASE_URL, timeout=300)
