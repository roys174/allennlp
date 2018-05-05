#!/usr/bin/env bash

f=/tmp/config.json

scripts/env_to_json.py - $f

python -m allennlp.run train -s /output/ $f

/bin/rm $f