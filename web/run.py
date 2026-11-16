"""Start the RoadSense web system:  python run.py"""
import os

from roadsense import create_app

app = create_app()

if __name__ == "__main__":
    # Use host="0.0.0.0" only if other devices on the network need to open the site.
    host = os.environ.get("ROADSENSE_HOST", "127.0.0.1")
    port = int(os.environ.get("ROADSENSE_PORT", "5000"))
    app.run(host=host, port=port, debug=False, use_reloader=True)
