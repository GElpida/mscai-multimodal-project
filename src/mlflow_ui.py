"""
Launch MLflow UI with the Windows SSL cert-store patch applied.

Usage (from repo root, with lavis_clean active):
    python src/mlflow_ui.py
    python src/mlflow_ui.py --port 5001
"""

import contextlib
import ssl
import sys

# Patch BEFORE any import touches ssl.create_default_context.
# aiohttp calls it at module level; if the patch isn't in place first, it crashes.
def _patched_load_windows_store_certs(self, storename, purpose):
    with contextlib.suppress(PermissionError):
        for cert, encoding, trust in ssl.enum_certificates(storename):
            if encoding == "x509_asn" and (trust is True or purpose.oid in trust):
                with contextlib.suppress(ssl.SSLError):
                    self.load_verify_locations(cadata=cert)

ssl.SSLContext._load_windows_store_certs = _patched_load_windows_store_certs

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Default args if none supplied.
if len(sys.argv) == 1:
    sys.argv += ["ui", "--backend-store-uri", "sqlite:///outputs/mlruns.db"]

from mlflow.cli import cli
cli()
