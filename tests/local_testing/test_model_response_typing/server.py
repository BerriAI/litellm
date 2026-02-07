# #### What this tests ####
# #    This tests if the litellm model response type is returnable in a flask app

# import sys, os
# import traceback
# from flask import Flask, request, jsonify, abort, Response
# sys.path.insert(0, os.path.abspath('../../..'))  # Adds the parent directory to the system path

# import litellm
# from litellm import completion

# litellm.set_verbose = False

# app = Flask(__name__)

# @app.route('/')
# def hello():
#     data = request.json
#     return completion(**data)

# if __name__ == '__main__':
#     from waitress import serve
#     serve(app, host='localhost', port=8080, threads=10)
