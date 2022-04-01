"""
Server for accessing spaCy's NLU features from a web application.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer

import spacy
from spacy import displacy

nlp = spacy.load("spacy_ro")

PORT_NUMBER = 3333


class RequestHandler(BaseHTTPRequestHandler):

    def parse(self, phrase):
        doc = nlp(phrase)

        options = {"add_lemma": True, "compact": True, "fine_grained": False}

        htmlDep = displacy.render(doc, style="dep", page=True, options=options)
        with open("dep-parse.html", "w", encoding='utf8') as f:
            f.write(htmlDep)

    # Handler for the POST requests
    def do_POST(self):
        self.send_response(200)

        content_len = int(self.headers.get('Content-Length'))
        phrase = self.rfile.read(content_len)

        phrase = phrase.decode("utf-8")
        print(phrase)
        self.parse(phrase)

        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        # Send the html message
        self.wfile.write(b"Hello World !")
        return


try:
    # Create a web server and define the handler to manage the incoming request
    server = HTTPServer(('', PORT_NUMBER), RequestHandler)
    print('Started httpserver on port ', PORT_NUMBER)

    # Wait forever for incoming htto requests
    server.serve_forever()

except KeyboardInterrupt:
    server.socket.close()
