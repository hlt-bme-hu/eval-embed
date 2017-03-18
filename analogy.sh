#!/bin/bash

MODEL_FILE="$1"
VOCAB_FILE="$2"

if [[ -z $3 ]]
then
    MODEL_TYPE=word2vec_text
else
    MODEL_TYPE=$3
fi

#see http://stackoverflow.com/questions/59895/can-a-bash-script-tell-which-directory-it-is-stored-in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ -z "$ANALOGY_FILE" ]]
then
    ANALOGY_FILE=$DIR/data/google_analogy_tests
fi
questions=`cat "$ANALOGY_FILE" | wc -l`

ARGS=("$@")

cat "$ANALOGY_FILE" | python2 $DIR/evaluate.py -m "$MODEL_FILE" \
-t $MODEL_TYPE -v "$VOCAB_FILE" -l 0 -n 1 -d cos -b 100 "${ARGS[@]:3}" | \
paste -d' ' "$ANALOGY_FILE.answers" - | awk '$1 == $2 { print $1}' | wc -l
