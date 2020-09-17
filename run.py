from acetumor import create_app

host = "0.0.0.0" 
port = 80
app = create_app()

if __name__ == '__main__':
    app.run(host=host, port=port)
