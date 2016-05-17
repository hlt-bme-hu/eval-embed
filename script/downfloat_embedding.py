#!/usr/bin/python
# vim: set fileencoding=utf-8 :

"""Downscales embeddings from float64 to float32 or float16."""

from argparse import ArgumentParser
import re

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

    for line in instream:
        fields = line.split()
        word, vector = fields[0], [float_type(s) for s in fields[1:]]
        print '{} {}'.format(word, ' '.join(map(str, vector)))
