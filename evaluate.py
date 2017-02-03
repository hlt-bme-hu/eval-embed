import sys
import numpy
import re
from scipy.sparse import lil_matrix, csr_matrix
import scipy.spatial.distance
import itertools
import struct
from multiprocessing import Process, Pipe
import argparse

# TODO prompt, tab completion, history

def renormalize(X):
    return X / (numpy.linalg.norm(X, axis=1)[:, None])

def renormalize_inplace(X):
    X /= (numpy.linalg.norm(X, axis=1)[:, None])

"""
define various distances here
don't forget to fill the 'compute_distance_table'
"""
def compute_cos(C, X, Y):
    return (C.dot(X)).dot(Y)

def compute_cos_r(C, X, Y):
    return renormalize(C.dot(X)).dot(Y)

def compute_cos_mul(C, X, Y):
    return C.dot(numpy.log1p(X.dot(Y)))
    
def compute_cos_mul1(C, X, Y):
    return C.dot(numpy.log(2 + X.dot(Y)))

def compute_cos_mul0(C, X, Y):
    return C.dot(numpy.log(X.dot(Y)))

def compute_eucl(C, X, Y):
    """
    TODO optimize this! It is ridiculous!
    """
    return scipy.spatial.distance.cdist(C.dot(X), Y.transpose())

def compute_eucl_mul(C, X, Y):
    return C.dot(numpy.log(scipy.spatial.distance.cdist(X, Y.transpose())))

def compute_arccos(C, X, Y):
    return C.dot(numpy.arccos(X.dot(Y)))

def compute_angle(C, X, Y):
    return (180.0/numpy.pi)*numpy.arccos(compute_cos_r(C, X, Y))

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
def glove_binary(vectors_fn, vocab_fn, fmt=float):
    word2index = read_vocab(vocab_fn)
    W = numpy.fromfile(vectors_fn, dtype=fmt).reshape((len(word2index), -1))
    return W, word2index

def glove_binary_bias(vectors_fn, vocab_fn, fmt=float):
    word2index = read_vocab(vocab_fn)
    W = numpy.fromfile(vectors_fn, dtype=fmt)
    W = W.reshape((len(word2index), -1))[:, :-1]
    return W, word2index

def glove_binary_context(vectors_fn, vocab_fn, fmt=float):
    word2index = read_vocab(vocab_fn)
    V = len(word2index)
    W = numpy.fromfile(vectors_fn, dtype=fmt).reshape((2*V, -1))
    W = W[:V, :] + W[V:, :]
    return W, word2index

def glove_binary_context_bias(vectors_fn, vocab_fn, fmt=float):
    word2index = read_vocab(vocab_fn)
    V = len(word2index)
    W = numpy.fromfile(vectors_fn, dtype=fmt).reshape((2*V, -1))
    W = W[:V, :-1] + W[V:, :-1]
    return W, word2index

def glove_text(vectors_fn, vocab_fn="", fmt=float):
    W = []
    word2index = {}
    for i, line in enumerate(open(vectors_fn, "r")):
        line = line.strip("\r").strip("\n").split(' ')
        word2index[line[0]] = i
        W.append(map(fmt, line[1:]))
    W = numpy.array(W)
    return W, word2index

def word2vec_text(vectors_fn, vocab_fn="", fmt=float):
    word2index = {}
    W = []
    f = open(vectors_fn, "r")
    V, dim = map(int, f.readline().strip().split(' '))[:2]
    W = numpy.zeros((V, dim), dtype=fmt)
    for i, line in enumerate(f):
        line = line.strip("\r").strip("\n").split(' ')
        word2index[line[0]] = i
        W[i, :] = map(fmt, line[1:])
    return W, word2index

def word2vec_binary(vectors_fn, vocab_fn="", fmt=numpy.float32):
    word2index = {}
    f = open(vectors_fn, "rb")
    V, dim = map(int, f.readline().strip().split(' '))
    struct_format = struct.Struct(("d" if fmt == float else "f")*dim)
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

def process_questions(C, words, all_words, n):
    scores = dist_function[0](C[:, all_words], W[all_words, :], W2)
    worst = -dist_function[1]*numpy.Inf
    for k in xrange(len(words)):
        scores[k, words[k]] = worst
    if dist_function[1] > 0:
        hits = scores.argpartition(-n, axis=1)[:, -n:]
        answers = [sorted(hits[i], key=lambda hit: scores[i, hit], reverse=True) for i in range(len(hits))]
    else:
        hits = scores.argpartition(n, axis=1)[:, :n]
        answers = [sorted(hits[i], key=lambda hit: scores[i, hit], reverse=False) for i in range(len(hits))]
    if args.log_level > 1:
        small_scores = [scores[i, answers[i]] for i in xrange(hits.shape[0])]
    else:
        small_scores = None
    return answers, small_scores

def input_process(input_queue_recv, output_queue_send):
    X = input_queue_recv.recv()
    while len(X) == 3:
        C, words, word_set = X
        answers, small_scores = process_questions(C, words, word_set, args.n_best)
        output_queue_send.send((answers, small_scores))
        X = input_queue_recv.recv()
    output_queue_send.send((None,))
    output_queue_send.close()
    
def output_process(output_queue_recv):
    X = output_queue_recv.recv()
    while len(X) == 2:
        answers, small_scores = X
        if small_scores is not None:
            for answer, score in zip(answers, small_scores):
                print args.answer_tab.join(
                            index2word[a] + (args.score_format % s)
                                for a, s in zip(answer, score)
                            )
        else:
            for a in answers:
                print args.answer_tab.join(index2word[w] for w in a)
        X = output_queue_recv.recv()

if __name__ == "__main__":
    regex_str_ = "([+-]?(\d*(\.\d+)?))\s*([^-+\s]+)"
    split_regex = re.compile(regex_str_)
    
    compute_distance_table = {x[0]: ((eval("compute_" + x[0]),) + x[1:]) for x in [
            # function, similarity or distance, requires normalization
            ("cos", 1, True), ("cos_r", 1, True),
            ("eucl", -1, False), ("eucl_mul", -1, False),
            ("cos_mul", 1, True), ("cos_mul0", 1, True), ("cos_mul1", 1, True),
            ("angle", -1, True), ("arccos", -1, True)]}

    input_types = {x: eval(x) for x in [
            "glove_binary",
            "glove_binary_bias",
            "glove_binary_context",
            "glove_binary_context_bias",
            "glove_text",
            "word2vec_text",
            "word2vec_binary"]}
    
    parser = argparse.ArgumentParser(
        #formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("Script for evaluating analogy tasks on word embedding vectors.\n"
            "Author: Gabor Borbely, borbely@math.bme.hu\n\n"
            "KING - MAN + WOMAN = QUEEN\n\n"
            "Reads question from stdin and writes answers to stdout, debug info to stderr.\n"
            "The questions can be asked in the following manner (one question per line):\n"
            "a single word for similarity: \"frog\",\n"
            "analogy as linear combination: \"king-man+woman\",\n"
            "compound: \"china+river\" or \"china river\",\n"
            "any linear combinations: \"1.0einstein+0.5cat-0.1relativity\".\n"
            "the linear combinations are parsed according to the following regex:\n") + \
            regex_str_
        )

    parser.add_argument('-v', '--vocab', dest="vocab_name", type=str,
                    default="vocab.txt", metavar="filename",
                    help="vocabulary file name needed for glove binary format")
    parser.add_argument('-v2', '--vocab2', dest="vocab_name2", type=str,
                    default="", metavar="filename")

    parser.add_argument('-m', '--model', dest="model_name", type=str,
                    default="vectors.bin", metavar="filename",
                    help="word vectors file")
    parser.add_argument('-m2', '--model2', dest="model_name2", type=str,
                    default="", metavar="filename")
    
    parser.add_argument('-n', '--best', dest="n_best", type=int,
                    default=5, metavar="uint",
                    help="number of answers per question, " + \
                    "you can see the n-best possible answers ordered by relevance")
    
    parser.add_argument('-d', '--distance', dest="distance_type", type=str,
                    default="cos_r", metavar="function",
                    help=" ".join(compute_distance_table.keys()))
    
    parser.add_argument('-t', '--type', dest="input_type", type=str,
                    default="glove_binary", metavar="function",
                    help=" ".join(input_types.keys()))
    parser.add_argument('-t2', '--type2', dest="input_type2", type=str,
                    default="", metavar="function")

    parser.add_argument('-b', '--batch', dest="batch_size", type=int,
                    default=100, metavar="uint",
                    help="the questions are collected into batches " + \
                    "and answered together in order to utilize vectorization")
    
    parser.add_argument('-f', '--format', dest="score_format", type=str,
                    default=" (%.3f)", metavar="format",
                    help="the format of the scores (relevance values) after the answers")
    
    parser.add_argument('-a', '--answer', dest="answer_tab", type=str,
                    default="\t", metavar="delimiter",
                    help="the delimiter between the answers")                    
    
    parser.add_argument('-l', '--log', dest="log_level", type=int,
                    default=1, choices=[0,1,2],
                    help="0: only answers to stdout, " + \
                    "1: parse info to stderr, " + \
                    "2: parse info and scores")

    parser.add_argument('-p', '--precision', dest="format", type=str,
                    default="None", metavar="constructor",
                    help="the floating point precision, You can set this to " + \
                    "float, numpy.float32, numpy.float64 or \"None\". " + \
                    "None means that the specific input type defines it.")

    parser.add_argument('-T', '--transform', dest="transform", type=str,
                    default="", metavar="filename",
                    help="Transformation matrix")

    args = parser.parse_args()

    dist_function = compute_distance_table[args.distance_type]
    input_function = input_types[args.input_type]
    input_function2 = input_function if args.input_type2 == "" else input_types[args.input_type2]
    
    if args.format is "None":
        # the specific input format tells which precision to use
        W, word2index = input_function(args.model_name, args.vocab_name)
        if args.model_name2 != "":
            W2, word2index2 = input_function2(args.model_name2, args.vocab_name2)
    else:
        W, word2index = input_function(args.model_name, args.vocab_name,
                                        eval(args.format))
        if args.model_name2 != "":
            W2, word2index2 = input_function2(args.model_name2, args.vocab_name2,
                                        eval(args.format))
    if args.model_name2 == "":
        W2, word2index2 = W, word2index
    
    index2word = {v: k for k, v in word2index2.iteritems()}
    
    if len(index2word) != len(word2index2):
        print >>sys.stderr, "You have multiple indices in word2index2!"
        exit(1)
    for embed, w2i in [(W, word2index), (W2, word2index2)]:
        if len(w2i) != len(embed):
            print >>sys.stderr, "You have", len(embed), "entries in the embedding"
            print >>sys.stderr, "but", len(w2i), " in the vocabulary!"
            exit(1)
        
    if args.transform != "":
        T = numpy.loadtxt(args.transform)
        if T.shape[0] != W.shape[1] or T.shape[1] != W2.shape[1]:
            print >>sys.stderr, "Transformation matrix dimension mismatch:", T.shape
            exit(1)
        W = W.dot(T)
        
    if dist_function[2]:
        renormalize_inplace(W2)
    
    # this duplicates the memory usage, but the metric computation is faster
    # W and W2.transpose() are both stored, even if W==W2
    W2 = W2.transpose()
    
    C = lil_matrix((args.batch_size, W.shape[0]), dtype=W.dtype)
    batch_index = 0
    words = []
    word_set = set()

    input_queue_recv, input_queue_send = Pipe()
    output_queue_recv, output_queue_send = Pipe()
    p = Process(target=input_process, args=(input_queue_recv, output_queue_send))
    p_out = Process(target=output_process, args=(output_queue_recv,))
    p.start()
    p_out.start()
    
    if args.log_level > 0:
        print >>sys.stderr, "parser ready"
    
    for line in sys.stdin:
        words.append([])
        this_words = words[-1]
        
        for term in split_regex.findall(line.strip()):
            if term[-1] in word2index:
                coeff = float(term[0] + ("1.0" if len(term[2]) == 0 else ""))
                word_id = word2index[term[-1]]
                this_words.append(word_id)
                word_set.add(word_id)
                C[batch_index, this_words[-1]] += coeff
                if args.log_level > 0:
                    print >>sys.stderr, coeff, "*", term[-1] + "\t",
        if args.log_level > 0:
            print >>sys.stderr
        batch_index += 1
        
        if batch_index >= args.batch_size:
            input_queue_send.send((
                C[:batch_index, :].tocsr(),
                words,
                list(word_set)))
            
            C = lil_matrix((args.batch_size, W.shape[0]), dtype=W.dtype)
            batch_index = 0
            words = []
            word_set = set()

    if batch_index > 0:
        input_queue_send.send((
            C[:batch_index, :].tocsr(),
            words,
            list(word_set)))
    
    input_queue_send.send((None,))
    input_queue_send.close()
    p.join()
    p_out.join()
