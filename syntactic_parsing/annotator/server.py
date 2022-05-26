"""
Server for accessing spaCy's NLU features from a web application.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import pathlib

import spacy
from spacy import displacy

nlp = spacy.load("ro")

dir = pathlib.Path(__file__).parent.resolve()
spacy_syntactic = spacy.load(dir / "../../models/spacy-syntactic-parser")

PORT_NUMBER = 3333


class RequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def parse(self, phrase):
        doc = nlp(phrase)

        options = {"add_lemma": True, "compact": True, "fine_grained": False}
        html_dep = displacy.render(doc, style="dep", page=True, options=options)

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(str.encode(html_dep))

    def store(self, new_example):
        with open(dir / "../data/t.json", encoding='utf-8') as examples_file:
            examples = examples_file.read()
        examples = json.loads(examples) if examples else []

        examples.append(json.loads(new_example))
        print(examples)

        with open(dir / "../data/t.json", "w", encoding='utf-8') as examples_file:
            examples_file.write(json.dumps(examples, ensure_ascii=False))

        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(str.encode(str(len(examples))))

    @staticmethod
    def _is_complex(sentence):
        doc = spacy_syntactic(sentence)
        return len([token for token in doc if token.dep_ == "ROOT"]) == 1 and len(doc) >= 3

    def do_GET(self):
        if self.path == "/next":
            with open(dir / "../data/pending_sentences.txt") as sentences_file:
                sentences = sentences_file.read().split("\n")

            with open(dir / "../data/t.json") as examples_file:
                examples = examples_file.read()
            examples = json.loads(examples)

            annotated_sentences = map(lambda e: e[0], examples)
            next_sentence = next(s for s in sentences if s not in annotated_sentences and self._is_complex(s))
            pending_sentences = list(filter(lambda s: s != next_sentence, sentences))

            with open(dir / "../data/pending_sentences.txt", "w") as sentences_file:
                sentences_file.write("\n".join(pending_sentences))

            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(str.encode(next_sentence))

    # Handler for the POST requests
    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        body = self.rfile.read(content_len)
        data = body.decode("utf-8")

        print(data)

        if self.path == "/dep":
            self.parse(data)
        elif self.path == "/store":
            self.store(data)
        return


try:
    # Create a web server and define the handler to manage the incoming request
    server = HTTPServer(('', PORT_NUMBER), RequestHandler)
    print('Started httpserver on port ', PORT_NUMBER)

    # Wait forever for incoming htto requests
    server.serve_forever()

except KeyboardInterrupt:
    server.socket.close()
