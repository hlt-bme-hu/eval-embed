import numpy
import argparse
import sys
import struct

from scipy.sparse import csr_matrix

class EmbeddingException(Exception):
    pass

def read_vocab(vocab_fn):
    word2index = {}
    with open(vocab_fn, "r") as f:
        for i, line in enumerate(f):
            word = line.strip().split()[0]
            word2index[word] = i
    return word2index


"""
define various formats here
don't forget to fill the input_types dictionary
"""
def read_dense_npz(vectors_fn, vocab_fn="", fmt="float64"):
    npz = numpy.load(vectors_fn)
    words = npz['words']
    W = npz['vectors']
    word2index = {w: i for i, w in enumerate(words)}
    return W, word2index

def read_csr_npz(vectors_fn, vocab_fn="", fmt="float64"):
    npz = numpy.load(vectors_fn)
    words = npz['words']
    W = csr_matrix((npz['vectors_data'], npz['vectors_indices'],
                    npz['vectors_indptr']), shape=npz['vectors_shape'])
    W = W.todense().A
    word2index = {w: i for i, w in enumerate(words)}
    return W, word2index

def read_glove_binary(vectors_fn, vocab_fn, fmt="float64"):
    word2index = read_vocab(vocab_fn)
    W = numpy.fromfile(vectors_fn, dtype=fmt).reshape((len(word2index), -1))
    return W, word2index

def read_glove_binary_bias(vectors_fn, vocab_fn, fmt=float):
    word2index = read_vocab(vocab_fn)
    W = numpy.fromfile(vectors_fn, dtype=fmt)
    W = W.reshape((len(word2index), -1))[:, :-1]
    return W, word2index

def read_glove_binary_context(vectors_fn, vocab_fn, fmt=float):
    word2index = read_vocab(vocab_fn)
    V = len(word2index)
    W = numpy.fromfile(vectors_fn, dtype=fmt).reshape((2*V, -1))
    W = W[:V, :] + W[V:, :]
    return W, word2index

def read_glove_binary_context_bias(vectors_fn, vocab_fn, fmt=float):
    word2index = read_vocab(vocab_fn)
    V = len(word2index)
    W = numpy.fromfile(vectors_fn, dtype=fmt).reshape((2*V, -1))
    W = W[:V, :-1] + W[V:, :-1]
    return W, word2index

def read_glove_text(vectors_fn, vocab_fn="", fmt="float64"):
    f = open(vectors_fn, "r") if type(vectors_fn) is str else vectors_fn
    file_pos = f.tell()
    V = numpy.loadtxt(f, dtype=str, comments=None, usecols=(0,))
    word2index = {w: i for i, w in enumerate(V)}
    f.seek(file_pos)
    W = numpy.loadtxt(f, dtype=fmt, comments=None,
                      converters={0: lambda x: 0.0})
    if len(word2index) != len(W):
        print >>sys.stderr, "WARNING:", len(word2index),
        print >>sys.stderr, "disjoint words in embedding of length", len(W)
    return W[:, 1:], word2index

def write_glove_text(W, word2index, vectors_fn, vocab_fn="", fmt="float64"):
    index2word = {i: w for w, i in word2index.items()}
    f = open(vectors_fn, "w") if type(vectors_fn) is str else vectors_fn
    # see https://en.wikipedia.org/wiki/Single-precision_floating-point_format
    # and https://en.wikipedia.org/wiki/Double-precision_floating-point_format
    if fmt == "float64":
        fmt = "%.17g"
    elif fmt == "float32":
        fmt = "%.9g"
    else:
        fmt = "%f"

    for i in range(len(W)):
        f.write(index2word[i] + " ")
        numpy.savetxt(f, W[i].reshape((1, -1)), fmt=fmt)

def read_word2vec_text(vectors_fn, vocab_fn="", fmt="float32"):
    f = open(vectors_fn, "r")
    V, dim = map(int, f.readline().strip().split())[:2]
    return read_glove_text(f, "", fmt)

def write_word2vec_text(W, word2index, vectors_fn, vocab_fn="", fmt="float32"):
    f = open(vectors_fn, "w")
    print >>f, W.shape[0], W.shape[1]
    return write_glove_text(W, word2index, f, "", fmt)

def read_word2vec_binary(vectors_fn, vocab_fn="", fmt="float32"):
    word2index = {}
    f = open(vectors_fn, "rb")
    V, dim = map(int, f.readline().strip().split()[:2])
    struct_format = struct.Struct(("d" if fmt == "float64" else "f")*dim)
    W = numpy.zeros((V, dim), dtype=fmt)
    for i in xrange(V):
        c = f.read(1)
        word = ''
        while c != ' ':
            word += c
            c = f.read(1)
        word = word.strip()
        word2index[word] = i
        W[i, :] = struct_format.unpack(f.read(struct_format.size))
    return W, word2index


"""
"""

input_types = {x: eval("read_" + x) for x in [
    "glove_binary", "glove_text",
    "glove_binary_bias",
    "glove_binary_context",
    "glove_binary_context_bias",
    "word2vec_text", "word2vec_binary",
    "csr_npz", "dense_npz"]}

output_types = {x: eval("write_" + x) for x in [
    "glove_text", "word2vec_text"]}

def main(args):
    print >>sys.stderr, "Reading ...",
    params = [args.model_from, args.vocab_from]
    if args.source_fmt != "None":
        params.append(args.source_fmt)

    W, word2index = input_types[args.input_type](*params)
    print >>sys.stderr, "Done"

    print >>sys.stderr, W.shape[0], W.shape[1]
    print >>sys.stderr, "Writing ...",
    params = [W, word2index, args.model_to, args.vocab_to]
    if args.target_fmt != "None":
        params.append(args.target_fmt)
    output_types[args.output_type](*params)
    print >>sys.stderr, "Done"
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converting between embedding formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-v1', '--vocab1', dest="vocab_from", type=str,
                        default="vocab.txt", metavar="filename",
                        help="vocabulary file name of source embedding")
    parser.add_argument('-v2', '--vocab2', dest="vocab_to", type=str,
                        default="vocab2.txt", metavar="filename",
                        help="vocabulary file name of target embedding")

    parser.add_argument('-m1', '--model1', dest="model_from", type=str,
                        default="vectors.txt", metavar="filename",
                        help="model file name of source embedding")
    parser.add_argument('-m2', '--model2', dest="model_to", type=str,
                        default="vectors2.txt", metavar="filename",
                        help="model file name of target embedding")

    parser.add_argument('-t1', '--type1', dest="input_type", type=str,
                        default="glove_binary", metavar="function",
                        help="source format: " + ", ".join(
                            sorted(input_types)),
                        choices=sorted(input_types))

    parser.add_argument('-t2', '--type2', dest="output_type", type=str,
                        default="word2vec_text", metavar="function",
                        help="target format: " + ", ".join(
                            sorted(output_types)),
                        choices=sorted(output_types))

    parser.add_argument('-f1', '--format1', dest="source_fmt", type=str,
                        default="None", metavar="precision",
                        help="source floating point precision, " +
                             "for \"None\" it is defined by specific " +
                             "embedding format.",
                        choices=["None", "float32", "float64"])
    parser.add_argument('-f2', '--format2', dest="target_fmt", type=str,
                        default="None", metavar="precision",
                        help="target floating point precision",
                        choices=["None", "float32", "float64"])

    args = parser.parse_args()
    exit(main(args))
