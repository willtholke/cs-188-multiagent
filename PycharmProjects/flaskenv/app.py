from flask import Flask
app = Flask(__name__)  # creating an application object as an instance of the class Flask


@app.route('/')  # http://127.0.0.1:5000/
def index():
    """ Index page. """
    return '<h1>well hello there</h1>'


# @app.route('/info')  # http://127.0.0.1:5000/info
# def info():
#     """ Information page. """
#     return '<p>Contact info:</p>' \
#            '<p>(650)862-4203</p>'
#
#
# @app.route('/puppy/<name>')  # http://127.0.0.1:5000/puppy/wanke
# def puppy(name):
#     """ Display puppy's name. """
#     return "100th letter: {}".format(name[100])


if __name__ == '__main__':
    app.run(debug=True)
