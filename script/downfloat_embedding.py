#!/usr/bin/python
# vim: set fileencoding=utf-8 :

"""Downscales embeddings from float64 to float32 or float16."""

from argparse import ArgumentParser
import re
import sys

import numpy as np


def parse_arguments():
    floatp = re.compile('float(\d+)')
    floats = filter(floatp.match, dir(np))
    float_map = {int(f[len('float'):]): getattr(np, f) for f in floats}
    default_float = {v: k for k, v in float_map.iteritems()}[np.float_]

    parser = ArgumentParser(
        description='Downscales (glove) embeddings from float64 to float32 or float16.')
    parser.add_argument('instream', metavar='embedding file', type=open,
                        help='The input embedding file.')
    parser.add_argument('--bits', '-b', type=int, default=default_float,
                        choices=sorted(float_map.keys()),
                        help='The number of float bits. The default is {}.'.format(
                            default_float))
    args = parser.parse_args()

    return args.instream, float_map[args.bits]


if __name__ == '__main__':
    instream, float_type = parse_arguments()

    diffs = []
    for line in instream:
        fields = line.split()
        word, vector = fields[0], np.array([s for s in fields[1:]],
                                           dtype=np.float_)
        down_vector = np.array(vector, dtype=float_type)
        diffs.append(np.linalg.norm(vector - down_vector))
        print '{} {}'.format(word, ' '.join(map(str, down_vector)))
    diffs = np.array(diffs, dtype=np.float_)
    print >>sys.stderr, 'Euclidean distance to the original:'
    print >>sys.stderr, '  - mean:', np.mean(diffs)
    print >>sys.stderr, '  - std:', np.std(diffs)
