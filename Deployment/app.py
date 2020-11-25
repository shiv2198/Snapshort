 
from flask import Flask
 
UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = 'secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 2048 * 2048


