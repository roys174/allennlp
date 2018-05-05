#!/usr/bin/env bash

f=/tmp/f.json

scripts/env_to_json.py - $f

python -m allennlp.run train -s /output/ $f

/bin/rm $f