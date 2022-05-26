# coding: utf-8
"""
Using the parser to recognise your own semantics
spaCy's parser component can be used to trained to predict any type of tree
structure over your input text. You can also predict trees over whole documents
or chat logs, with connections between the sentence-roots used to annotate
discourse structure. In this example, we'll build a message parser for a common
"chat intent": finding local businesses. Our message semantics will have the
following types of relations: ROOT, PLACE, QUALITY, ATTRIBUTE, TIME, LOCATION.
"show me the best hotel in berlin"
('show', 'ROOT', 'show')
('best', 'QUALITY', 'hotel') --> hotel with QUALITY best
('hotel', 'PLACE', 'show') --> show PLACE hotel
('berlin', 'LOCATION', 'hotel') --> hotel with LOCATION berlin
Compatible with: spaCy v2.0.0+
"""

import plac
import random
from pathlib import Path
import spacy
from spacy.tokens import Span
from spacy.util import minibatch, compounding
from spacy import displacy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from utils import TermColors
import json
import yaml
import itertools

with open("dependencies.txt") as dependencies_file:
    dependency_types = list(map(lambda line: line.strip(), dependencies_file.readlines()))

# data examples: texts, heads and dependency labels
# for no relation, we simply chose an arbitrary dependency label, e.g. '-'
with open("data/train.json") as data_file:
    train_data = json.load(data_file)

with open("data/test.json") as data_file:
    test_data = json.load(data_file)


def analyze_data(phrases):
    """
    Print statistics about the given phrases
    :param phrases: list of phrases to analyze
    """

    dep_freq = {}
    for phrase, relations in phrases:
        for dep in relations['deps']:
            dep_freq[dep] = dep_freq.get(dep, 0) + 1

    dep_freq = {k: v for k, v in sorted(dep_freq.items(), key=lambda item: item[1])}
    print('Depencencies frequencies:')
    print(dep_freq)

    print("TRAIN EXAMPLES: ", len(train_data))

    y_pos = np.arange(len(dependency_types))

    plt.barh(y_pos, [dep_freq[q] for q in dependency_types], align='center', alpha=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(y_pos, dependency_types, fontsize=12)
    plt.xlabel('Number of occurrences', fontsize=13)
    plt.ylabel('Syntactic question (label)', fontsize=13)
    plt.title('Syntactic question frequencies in the train examples', fontsize=13)
    plt.savefig("../results/syntactic_questions_freq.svg", format="svg", bbox_inches='tight')


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    n_iter=("Number of training iterations", "option", "n", int),
)
def train(model=None, n_iter=30):
    """ Load the model, set up the pipeline and train the parser. """

    print("Loading model...")
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("ro")  # create blank Language class
        print("Created blank 'ro' model")

    # We'll use the built-in dependency parser class, but we want to create a fresh instance – just in case.
    # if "parser" in nlp.pipe_names:
    #     nlp.remove_pipe("parser")
    # parser = nlp.create_pipe("parser")
    # nlp.add_pipe(parser, first=True)
    parser = [pipe for (name, pipe) in nlp.pipeline if name == "parser"][0]

    for text, annotations in train_data:
        for dep in annotations.get("deps", []):
            parser.add_label(dep)

    pipe_exceptions = ["parser", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()

        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            print(itn, "Losses", losses)

    return nlp


def dep_span(doc, token, span_level=0):
    def dfs(node):
        first = last = node.i
        for child in node.children:
            if child.dep_ in ['-'] or \
                    (span_level >= 1 and child.dep_ == "prep") or \
                    (span_level >= 2 and child.dep_ in ['care', 'ce fel de', 'cât', 'al cui']):
                child_first, child_last = dfs(child)
                first = min(first, child_first)
                last = max(last, child_last)
        return first, last

    first, last = dfs(token)  # compute bounds of the span
    span = Span(doc, first, last + 1)
    return span.text


def print_parsing_result(doc):
    for token in doc:
        if token.dep_ != "-" and token.dep_ != 'prep':
            print(TermColors.YELLOW, token.dep_, TermColors.ENDC,
                  f'[{dep_span(doc, token.head, 0)}] ->',
                  TermColors.RED, dep_span(doc, token, 2), TermColors.ENDC)


def test_model(nlp, interactive=False):
    """
    Test the results of the model for a set of predefined sentences
    or by interactively introducing sentences to the prompt.
    """

    with open("data/test_blank.txt") as test_file:
        lines = map(lambda line: line.strip(), test_file.readlines())
        texts = filter(lambda line: line, lines)

    # Load test examples from the conversational agent dataset
    with open("../microworlds/test/mem_assistant/test_nlu.yml") as stream:
        try:
            content = yaml.safe_load(stream)
            per_intent_examples = map(lambda intent_entry: intent_entry['examples'].split('\n'), content['nlu'])
            all_examples = filter(lambda ex: ex, itertools.chain.from_iterable(per_intent_examples))

            conv_agent_test_sentences = list(map(lambda per_intent_ex: per_intent_ex.strip("- "), all_examples))
        except yaml.YAMLError as e:
            print(e)

    if interactive:
        print("\nInteractive testing. Enter a phrase to parse it:")
        while True:
            phrase = input("\n>> ")
            doc = nlp(phrase)
            print_parsing_result(doc)
    else:
        docs = nlp.pipe(conv_agent_test_sentences)
        for doc in docs:
            print('\n', doc.text)
            print_parsing_result(doc)

        # show a visual result as a web page with the help of displacy
        docs = [nlp(phrase) for phrase in texts]
        options = {"add_lemma": False, "compact": True, "fine_grained": False}

        html_dep = displacy.render(docs, style="dep", page=True, options=options)
        with open("deps.html", "w", encoding='utf8') as f:
            f.write(html_dep)


def plot_confusion_matrix(y_true, y_pred, labels):
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

    sns.heatmap(conf_mat, annot=conf_mat, fmt='g', cmap='Greens',
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 11})
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title("Syntactic questions prediction", fontsize=14)
    plt.savefig("../results/syntactic_questions_pred.svg", format="svg", bbox_inches='tight')


def evaluate_model(nlp):
    """ Compute evaluation metrics (confusion matrix, classification report, accuracy) for the model. """

    deps_true = []
    deps_pred = []
    correct_heads = {dep: 0 for dep in dependency_types}
    num_deps = {dep: 0 for dep in dependency_types}

    # parse sentences
    docs = nlp.pipe(map(lambda s: s[0], test_data))

    # evaluate predictions
    for i, doc in enumerate(docs):
        true_sentence_deps = test_data[i][1]

        # evaluate dependencies (syntactic questions) prediction
        sentence_deps_true = true_sentence_deps['deps']
        sentence_deps_pred = [token.dep_ for token in doc]

        deps_true += sentence_deps_true
        deps_pred += sentence_deps_pred

        if any(true != pred for (true, pred) in zip(sentence_deps_true, sentence_deps_pred)):
            print()
            print(sentence_deps_true)
            print(sentence_deps_pred)

            for token in doc:
                if token.dep_ != "-":
                    print(TermColors.YELLOW, token.dep_, TermColors.ENDC, f'[{token.head.text}] ->',
                          TermColors.GREEN, token.text, TermColors.ENDC)

        # evaluate heads prediction
        for j, token in enumerate(doc):
            if token.head.i == true_sentence_deps['heads'][j]:
                correct_heads[true_sentence_deps['deps'][j]] += 1
            num_deps[true_sentence_deps['deps'][j]] += 1

    print("Number of test examples", len(test_data))

    print("Syntactic questions accuracy:")
    print(classification_report(deps_true, deps_pred, zero_division=0))
    plot_confusion_matrix(deps_true, deps_pred, dependency_types)

    print("Heads accuracy: ", sum(correct_heads.values()) / sum(num_deps.values()), '\n')

    for dep in dependency_types:
        acc = round(correct_heads[dep] / num_deps[dep], 2) if num_deps[dep] > 0 else '-'
        print(f'- {dep}: {" " * (15 - len(dep))}{acc}')


def main():
    analyze_data(train_data)

    # uncomment this to train the model before the testing stage
    model = train("ro", n_iter=15)
    model.to_disk(Path('../models/spacy-syntactic-parser'))

    nlp = spacy.load('../models/spacy-syntactic-parser')

    # evaluate the model using a separate test set
    evaluate_model(nlp)

    # test the model for debugging
    # test_model(nlp, interactive=False)


if __name__ == "__main__":
    main()
