# eval-embed
Conversions between embedding formats &amp; evaluations against industry standard datasets.

## General Evaluate

    python2 evaluate.py -h

This scripts is for various embedding evaluation tasks, such as:
* analogy task
* linear translation

### Known embedding formats
* GloVe text and binary with optional bias and context terms
* word2vec text and binary

The list can be expanded, you can easily write your custom input format.

For GloVe binary vectors also the corresponding vocabulary should be provided. Floating point precision 32 or 64 bit can be used.

### Questions
The script reads _questions_ from stdin and answers them, line by line.
The answers are written to stdout, additional debug info to stderr.
A question can be any linear combination of input terms, such as:
* king -man +woman
* frog
* chinese + river

### Possible evaluation metrics
* `cos`: standard cosine similarity
* `cos_r`: the answers are the same as with cosine similarity, but the true similarity of the outcomes is shown. If you do not care the similarity scores, just the answers, then you can use plain `cos` because it is slightly faster.
* `eucl`: square of the standard Euclidean metric
* `eucl_r`: the standard Euclidean metric
* `eucl_norm`: Euclidean metric but vectors are normalized first. This should be the same as `cos` or `cos_r`
* `cos_mul`: the so called cos-mul metric, used in analogy tasks
* `cos_mul0`: per default, cos-mul operates on (1+cos) since the cos similarity varies from -1 ot 1. But with mul0, positive vectors are assumed.
* `arccos`: arc length distance on the unit sphere
* `eucl_mul`: mutiplicative Euclidean
* `angle`: same as cosine similarity but you can see the actual angle

The list can be expanded, you can easily write your custom metric or similarity.

### Translation
Using the `translate.py` script, you can generate a linear transformation between embeddings.
If you have two embeddings and a transformation between them, then you can query in one language and retrieve answers in the other.
Lets say you have an English source and a German target embedding and a linear transformation matrix, then:

    king - man + woman = Königin

### Dependencies
* numpy
* scipy
