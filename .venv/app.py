import Smile as Backend
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__, static_url_path="/static", static_folder="static")

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return render_template("index.html", title="Home")

@app.route('/sendImage', methods=['POST'])
def sent():
    if request.method == "POST":
        # Take payload, convert it to image we're expecting
        request_data = request.get_json()
        print(request_data)
        # Run through backend
        # For now, save images made in folder
    return "hello world"
# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()