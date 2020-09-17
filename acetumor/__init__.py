from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "invisiblevision"
    app.config["UPLOADED_IMAGES"] = "uploads/images"
    from acetumor.api import api
    app.register_blueprint(api, url_prefix="/api")
        
    return app