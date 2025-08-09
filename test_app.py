import requests
import json

def test_flask_app():
    """Test the Flask application"""
    try:
        # Test if the app is running
        response = requests.get('http://localhost:5000')
        if response.status_code == 200:
            print("✅ Flask app is running successfully!")
            print("🌐 Webpage is accessible at: http://localhost:5000")
        else:
            print(f"❌ Flask app returned status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Flask app is not running. Please start it with: python app.py")
    except Exception as e:
        print(f"❌ Error testing Flask app: {e}")

if __name__ == "__main__":
    test_flask_app()
