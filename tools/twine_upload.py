"""Run twine with the operating system's certificate store.

Needed on machines where antivirus HTTPS scanning or a corporate proxy causes
"SSL: CERTIFICATE_VERIFY_FAILED" with Python's bundled certificates.
Certificate verification stays ON; only the trust store changes.

Usage (same arguments as twine):
    python tools/twine_upload.py upload --repository testpypi dist/*
    python tools/twine_upload.py upload dist/*

twine asks for the API token itself; this script never reads or stores it.
Not shipped with the package (development helper only).
"""

import sys

import truststore

truststore.inject_into_ssl()

from twine.__main__ import main  # noqa: E402  (must import after inject_into_ssl)

if __name__ == "__main__":
    sys.argv[0] = "twine"
    sys.exit(main())
