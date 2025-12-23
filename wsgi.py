"""
WSGI entry point for Vercel deployment
"""

from app import app

# Vercel expects the application object to be named 'app' or 'application'
# This file ensures that Flask app is properly imported and accessible
if __name__ == "__main__":
    app.run()