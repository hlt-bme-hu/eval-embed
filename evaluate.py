import sys
import numpy
import re
import scipy.sparse
import itertools
from multiprocessing import Process, Pipe
import argparse
from convert import input_types
import readline

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

def compute_proj(C, X, Y):
    return C.dot(numpy.abs(X.dot(Y)))
    
def compute_cos_mul(C, X, Y):
    return C.dot(numpy.log1p(X.dot(Y)))
    
def compute_cos_mul1(C, X, Y):
    return C.dot(numpy.log(2 + X.dot(Y)))

def compute_cos_mul0(C, X, Y):
    return C.dot(numpy.log(X.dot(Y)))

def euclidean_distances(X, Y):
    """
    http://stackoverflow.com/questions/6430091/efficient-distance-calculation-between-n-points-and-a-reference-in-numpy-scipy
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html#sklearn.metrics.pairwise.euclidean_distances
    """
    return W_norms[None, :] - 2 * X.dot(Y)

def euclidean_distances_r(X, Y):
    return numpy.sqrt((X**2).sum(axis=1)[:, None] - 2 * X.dot(Y) + W_norms[None, :])

def compute_eucl(C, X, Y):
    return euclidean_distances(C.dot(X), Y)
    
def compute_eucl_r(C, X, Y):
    return euclidean_distances_r(C.dot(X), Y)
    
def compute_eucl_norm(C, X, Y):
    return compute_eucl(C, X, Y)

def compute_eucl_mul(C, X, Y):
    return C.dot(numpy.log(scipy.spatial.distance.cdist(X, Y.transpose())))

def compute_arccos(C, X, Y):
    return C.dot(numpy.arccos(X.dot(Y)))

def compute_angle(C, X, Y):
    return (180.0/numpy.pi)*numpy.arccos(compute_cos_r(C, X, Y))

def get_stdin():
    """
    Get data from stdin, if any
    """
    if not sys.stdin.isatty():
        for line in sys.stdin:
            yield line
    return

def process_questions(C, all_words, n):
    scores = dist_function[0](C[:, all_words], W[all_words, :], W2)
    worst = -dist_function[1]*numpy.Inf
    for i in range(C.shape[0]):
        scores[i, C[i, :].nonzero()[1]] = worst
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
    while len(X) == 2:
        C, word_set = X
        answers, small_scores = process_questions(C, word_set, args.n_best)
        output_queue_send.send((C if args.log_level > 0 else None, 
                                answers, small_scores))
        X = input_queue_recv.recv()
    output_queue_send.send((None,))
    output_queue_send.close()
    
def output_process(output_queue_recv):
    X = output_queue_recv.recv()
    while len(X) == 3:
        coeffs, answers, small_scores = X
        n = len(answers)
        for i in range(n):
            if coeffs is not None:
                coeffs_text = ["%g * %s" % (c, index2word1[w]) \
                            for w,c in zip(*scipy.sparse.find(coeffs[i])[1:])]
                print >>sys.stderr, args.answer_tab.join(coeffs_text)
            if small_scores is not None:
                score_text = [(args.score_format % s) for s in small_scores[i]]
                print args.answer_tab.join(
                                index2word[a] + s
                                    for a, s in zip(answers[i], score_text)
                                )
            else:
                print args.answer_tab.join(index2word[w] for w in answers[i])
        X = output_queue_recv.recv()

def stdin_reader():
    for line in sys.stdin:
        yield line.strip()
def prompt_reader():
    try:
        while True:
            yield raw_input('')
    except EOFError:
        pass

class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                        argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    regex_str_ = "([+-]?(\d*(\.\d+)?))\s*(\S+)\s*"
    split_regex = re.compile(regex_str_)
    
    compute_distance_table = {x[0]: ((eval("compute_" + x[0]),) + x[1:]) for x in [
            # function, similarity or distance, requires normalization
            ("cos", 1, True), ("cos_r", 1, True), ("proj", 1, True),
            ("eucl", -1, False), ("eucl_mul", -1, False), ("eucl_norm", -1, True),
            ("eucl_r", -1, False),
            ("cos_mul", 1, True), ("cos_mul0", 1, True), ("cos_mul1", 1, True),
            ("angle", -1, True), ("arccos", -1, True)]}
    
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=("Script for evaluating analogy tasks on word embedding vectors.\n"
            "Author: Gabor Borbely, borbely@math.bme.hu\n"
            "      __________________ _  \n"
            "     /                  / \ \n"
            "    { king -man +woman {  | \n"
            "    }                  }_/  \n"
            "    {         =        {    \n"
            "   _}                  {    \n"
            "  /@{       QUEEN      {    \n"
            " |  }                  }    \n"
            " \_/__________________/     \n\n"
            "Reads question from stdin and writes answers to stdout, debug info to stderr.\n"
            "The questions can be asked in the following manner (one question per line):\n"
            "a single word for similarity: \"frog\",\n"
            "analogy as linear combination: \"king - man + woman\",\n"
            "compound: \"china +river\" or \"china river\",\n"
            "any linear combinations: \"1.0einstein +0.5cat -0.1relativity\".\n"
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
                    help=", ".join(compute_distance_table.keys()))
    
    parser.add_argument('-t', '--type', dest="input_type", type=str,
                    default="glove_binary", metavar="function",
                    help=", ".join(input_types.keys()))
    parser.add_argument('-t2', '--type2', dest="input_type2", type=str,
                    default="", metavar="function")

    parser.add_argument('-b', '--batch', dest="batch_size", type=int,
                    default=100, metavar="uint",
                    help="the questions are collected into batches " + \
                    "and answered together in order to utilize vectorization")
    
    parser.add_argument('-f', '--format', dest="score_format", type=str,
                    default=" (%.3g)", metavar="format",
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
    
    params1 = [args.model_name, args.vocab_name]
    params2 = [args.model_name2, args.vocab_name2]
    if args.format != "None":
        params1.append(eval(args.format))
        params2.append(eval(args.format))
    W, word2index = input_function(*params1)
    if args.model_name2 != "":
        W2, word2index2 = input_function2(*params2)
    else:
        W2, word2index2 = W, word2index
    
    index2word = {v: k for k, v in word2index2.iteritems()}
    index2word1 = {v: k for k, v in word2index.iteritems()}
    
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
        # this renormalizes W and W2 if they point to the same object
        renormalize_inplace(W2)
    
    W_norms = (W2**2).sum(axis=1)
    
    # this duplicates the memory usage, but the cos similarity is slightly faster
    # W and W2.transpose() are both stored, even if W==W2
    W2 = W2.transpose()

    if sys.stdin.isatty():
        if args.log_level > 0:
            print >>sys.stderr, "parser ready"
        generator = prompt_reader()
        args.batch_size = 1
    else:
        generator = stdin_reader()

    C = scipy.sparse.lil_matrix((args.batch_size, W.shape[0]), dtype=W.dtype)
    batch_index = 0
    words = []
    word_set = set()

    input_queue_recv, input_queue_send = Pipe()
    output_queue_recv, output_queue_send = Pipe()
    p = Process(target=input_process, args=(input_queue_recv, output_queue_send))
    p_out = Process(target=output_process, args=(output_queue_recv,))
    p.start()
    p_out.start()

    for line in generator:
        words.append([])
        this_words = words[-1]
        
        for term in split_regex.findall(line.strip()):
            if term[-1] in word2index:
                coeff = float(term[0] + ("1.0" if len(term[2]) == 0 else ""))
                word_id = word2index[term[-1]]
                this_words.append(word_id)
                word_set.add(word_id)
                C[batch_index, this_words[-1]] += coeff
        batch_index += 1
        
        if batch_index >= args.batch_size:
            input_queue_send.send((C[:batch_index, :].tocsr(), list(word_set)))
            C = scipy.sparse.lil_matrix((args.batch_size, W.shape[0]), dtype=W.dtype)
            batch_index = 0
            words = []
            word_set = set()

    # left-overs
    if batch_index > 0:
        input_queue_send.send((C[:batch_index, :].tocsr(), list(word_set)))
    
    input_queue_send.send((None,))
    input_queue_send.close()
    p.join()
    p_out.join()
