#!/usr/bin/env bash

f=/tmp/config.json

python scripts/env_to_json.py - $JSON_FILE $ENCODER_FIELDS $f

python -m allennlp.run train -s /output $f

/bin/rm $f
