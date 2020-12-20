#!/usr/bin/env python3
# coding: utf8

"""
Load vectors for a language trained using fastText
https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
"""

import gzip
import numpy
import plac
import spacy

@plac.annotations(
    vectors_loc=("Path to .vec.gz file", "positional", None, str),
    lang=("Language identifier (e.g. 'ro')", "positional", None, str),
    output_dir=("Output directory", "positional", None, str))
def main(vectors_loc, lang=None, output_dir=None):
    nlp = spacy.blank(lang)

    with gzip.open(vectors_loc) as file:
        header = file.readline()
        nr_row, nr_dim = header.split()
        nlp.vocab.reset_vectors(width=int(nr_dim))

        for line in file:
            line = line.rstrip().decode('utf8')
            pieces = line.rsplit(' ', int(nr_dim))
            word = pieces[0]
            vector = numpy.asarray([float(v) for v in pieces[1:]], dtype='f')
            nlp.vocab.set_vector(word, vector)

    nlp.to_disk(output_dir)


if __name__ == '__main__':
    plac.call(main)
