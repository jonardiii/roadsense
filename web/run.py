"""Start the RoadSense web system:  python run.py"""
from roadsense import create_app

app = create_app()

if __name__ == "__main__":
    # Use host="0.0.0.0" only if other devices on the network need to open the site.
    app.run(host="127.0.0.1", port=5000, debug=False)
