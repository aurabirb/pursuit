"""Aiohttp web server for serving static files over HTTP and HTTPS."""

import os
import ssl
import subprocess
from pathlib import Path

from aiohttp import web

# Configuration (can be overridden by environment variables)
HOST = os.getenv("WEB_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("WEB_PORT", "8080"))
HTTPS_PORT = int(os.getenv("WEB_HTTPS_PORT", "8443"))
STATIC_DIR = Path(__file__).parent / "static"
CERTS_DIR = Path(__file__).parent / "certs"
CERT_FILE = CERTS_DIR / "localhost.crt"
KEY_FILE = CERTS_DIR / "localhost.key"


def generate_self_signed_cert() -> None:
    """Generate a self-signed certificate for localhost if it doesn't exist."""
    if CERT_FILE.exists() and KEY_FILE.exists():
        return

    CERTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating self-signed certificate for localhost...")
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:4096",
            "-keyout", str(KEY_FILE),
            "-out", str(CERT_FILE),
            "-days", "365",
            "-nodes",
            "-subj", "/CN=localhost",
            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    print(f"Certificate generated: {CERT_FILE}")


def create_ssl_context() -> ssl.SSLContext | None:
    """Create an SSL context for HTTPS, or None if certs unavailable."""
    try:
        generate_self_signed_cert()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not generate SSL certificate: {e}")
        print("HTTPS will not be available. Install openssl to enable HTTPS.")
        return None

    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(str(CERT_FILE), str(KEY_FILE))
    return ssl_context


async def index_handler(request: web.Request) -> web.FileResponse:
    """Serve index.html for the root path."""
    return web.FileResponse(STATIC_DIR / "index.html")


def create_app(static_dir: Path = None) -> web.Application:
    """Create and configure the aiohttp web application.

    Args:
        static_dir: Optional path to static files directory. Defaults to ./static/
    """
    static = static_dir or STATIC_DIR
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_static("/static/", static, name="static")
    return app


async def start_server(
    runner: web.AppRunner,
    host: str = None,
    http_port: int = None,
    https_port: int = None,
    ssl_context: ssl.SSLContext = None,
) -> None:
    """Start the web server asynchronously on both HTTP and HTTPS.

    Args:
        runner: The aiohttp AppRunner to use
        host: Host to bind to. Defaults to WEB_HOST env var or 0.0.0.0
        http_port: HTTP port. Defaults to WEB_PORT env var or 8080
        https_port: HTTPS port. Defaults to WEB_HTTPS_PORT env var or 8443
        ssl_context: SSL context for HTTPS. If None, HTTPS is disabled.
    """
    await runner.setup()
    h = host or HOST

    # Start HTTP site
    http_site = web.TCPSite(runner, h, http_port or HTTP_PORT)
    await http_site.start()
    print(f"HTTP server running at http://{h}:{http_port or HTTP_PORT}")

    ssl_context = ssl_context or create_ssl_context()

    # Start HTTPS site if SSL context is available
    if ssl_context:
        https_site = web.TCPSite(runner, h, https_port or HTTPS_PORT, ssl_context=ssl_context)
        await https_site.start()
        print(f"HTTPS server running at https://{h}:{https_port or HTTPS_PORT}")


def run_standalone() -> None:
    """Run the web server in standalone mode (blocking) on both HTTP and HTTPS."""
    import asyncio

    if not STATIC_DIR.exists():
        STATIC_DIR.mkdir(parents=True)
        print(f"Created static directory: {STATIC_DIR}")

    ssl_context = create_ssl_context()

    app = create_app()
    print(f"Serving static files from: {STATIC_DIR}")

    async def run_both():
        runner = web.AppRunner(app)
        await start_server(runner, HOST, HTTP_PORT, HTTPS_PORT, ssl_context)
        # Keep running
        while True:
            await asyncio.sleep(3600)

    try:
        asyncio.run(run_both())
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    run_standalone()
