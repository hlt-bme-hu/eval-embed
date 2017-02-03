import sys
import numpy
from collections import defaultdict
from itertools import chain
import argparse

def renormalize(M):
    return M / numpy.linalg.norm(M, axis=1)[:, None]

def renormalize_vector(v):
    return v / numpy.linalg.norm(v)

def outer(l1, l2):
    return list(chain(*[[(x,y) for x in l1] for y in l2]))
    
def read_embed(file, word_list):
    n, dim = map(int, file.readline().strip().split())
    W = []
    V = defaultdict(list)
    i2w = {}
    i = 0
    multi = False
    for line in file:
        parts = line.strip().split()
        if len(word_list) == 0 or parts[0] in word_list:
            W.append(map(float, parts[1:]))
            V[parts[0]].append(i)
            i2w[i] = parts[0]
            if not multi and len(V[parts[0]]) > 1:
                multi = True
            i += 1
    return numpy.array(W), dict(V), i2w, multi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("type", type=str, choices=["train", "test"])
    parser.add_argument("embed1")
    parser.add_argument("embed2")
    parser.add_argument("seed_dict")
    parser.add_argument("--train-mode", dest="train_mode", default="single",
                                choices=["single", "first", "all"])
    parser.add_argument("-n", dest="n", default=5, type=int,
                                help="number of examples shown")
    parser.add_argument("--verbose", help="writes translation examples to stderr",
                    action="store_true")
    args = parser.parse_args()
    
    seed_list = [tuple(line.strip().split()) for line in open(args.seed_dict, "r")]
    if args.type == "train":
        lang1_words = [pair[0] for pair in seed_list]
        lang2_words = [pair[1] for pair in seed_list]
    else:
        if args.verbose:
            lang1_words = []
            lang2_words = []
        else:
            lang1_words = [pair[0] for pair in seed_list]
            lang2_words = []
    
    W1, V1, i2w1, multi1 = read_embed(open(args.embed1), lang1_words)
    W2, V2, i2w2, multi2 = read_embed(open(args.embed2), lang2_words)
    
    if args.type == "train":
        M1 = numpy.zeros((0, W1.shape[1]))
        M2 = numpy.zeros((0, W2.shape[1]))
        
        if args.train_mode == "single":
            if multi1 or multi2:
                print >>sys.stderr, "Not a single prototype embedding!"
                exit(1)
        
        train_pairs = [(V1[s], V2[t]) for s, t in seed_list if s in V1 and t in V2]
        if args.train_mode == "first":
            train_pairs = [(p1[0], p2[0]) for p1, p2 in train_pairs]
        else:
            train_pairs = list(chain(*[outer(p1, p2) for p1, p2 in train_pairs]))
        lang1_indices, lang2_indices = zip(*train_pairs)
        
        M1 = W1[lang1_indices, :]
        M2 = W2[lang2_indices, :]
        
        T = numpy.linalg.lstsq(M1, M2)[0]
        numpy.savetxt(sys.stdout, T)
    else:
        T = numpy.loadtxt(sys.stdin)
        
        W2 = renormalize(W2)
        
        seed_dict = defaultdict(set)
        for source, target in seed_list:
            seed_dict[source].add(target)
        seed_dict = dict(seed_dict)
        
        for source, targets in seed_dict.iteritems():
            weak_hit = W2.shape[0]
            weak_answers = list(chain(*[V2[t] for t in targets if t in V2]))
            strong_hits = [W2.shape[0]] * len(targets)
            strong_answers = [V2[t] if t in V2 else [] for t in targets]
            if source in V1:
                for s in V1[source]:
                    translated = renormalize_vector(W1[s].dot(T))
                    scores = W2.dot(translated)
                    indices = numpy.argsort(scores)[::-1]
                    
                    if args.verbose:
                        closest = (numpy.argsort(W1.dot(W1[s]))[::-1])[:args.n]
                        for c in closest:
                            print >>sys.stderr, i2w1[c], 
                        print >>sys.stderr, "->", 
                        for t in indices[:args.n]:
                            print >>sys.stderr, i2w2[t], 
                        print >>sys.stderr, "|", 
                        for a in targets:
                            print >>sys.stderr, a,
                        print >>sys.stderr
                    
                    if len(weak_answers) > 0:
                        this_weak_hit = min(list(indices).index(t) for t in weak_answers)
                        if this_weak_hit < weak_hit:
                            weak_hit = this_weak_hit
                    for j in range(len(targets)):
                        if len(strong_answers[j]) > 0:
                            this_strong_hit = min(list(indices).index(t) for t in strong_answers[j])
                            if this_strong_hit < strong_hits[j]:
                                strong_hits[j] = this_strong_hit
            
            for strong_hit, target in zip(*[strong_hits, targets]):
                print weak_hit + 1, strong_hit + 1, source, target
