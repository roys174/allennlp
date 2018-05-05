#!/usr/bin/env bash

import sys
import json


def main(args):
    argc = len(args)

    if argc < 3:
        print("Usage:", args[0], "<env file> <json ofile>")
        return -1

    if args[1] == '-':
        print("Reading args from env variables")
        import os
        v = os.environ
    else:
        print("Reading args from", args[1])
        with open(args[1]) as ifh:
            v = dict([(l.split()[1].split("=")) for l in ifh])

    base = {
        "dataset_reader": {
            "type": "conll2003",
            "tag_label": "ner",
            "coding_scheme": "BIOUL",
            "token_indexers": {
                "tokens": {
                    "type": "single_id",
                    "lowercase_tokens": True
                },
                "token_characters": {
                    "type": "characters"
                }
            }
        },
        "train_data_path": "/input/eng.train",
        "validation_data_path": "/input/eng.testa",
        "test_data_path": "/input/eng.testb",
        "evaluate_on_test": True,
        "model": {
        "type": "crf_tagger",
        "constraint_type": "BIOUL",
        "dropout": 0.5,
        "include_start_end_transitions": False,
        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 50,
                "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
                "trainable": True
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": 16
                },
                "encoder": {
                    "type": "cnn",
                    "embedding_dim": 16,
                    "num_filters": 128,
                    "ngram_filter_sizes": [3],
                    "conv_layer_activation": "relu"
                }
            }
        },
        "regularizer": [
          [
            "scalar_parameters",
            {
              "type": "l2",
              "alpha": float(v["WEIGHT_DECAY"])
            }
          ]
        ]
      },
      "iterator": {
        "type": "basic",
        "batch_size": int(v["BATCH_SIZE"])
      },
      "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": float(v["LEARNING_RATE"])
        },
        "validation_metric": "+f1-measure-overall",
        "num_serialized_models_to_keep": 3,
        "num_epochs": int(v["N_EPOCHS"]),
        "grad_clipping": float(v["CLIP_GRAD"]),
        "patience": int(v["PATIENCE"]),
        "cuda_device": 0,
        "learning_rate_scheduler": {
              "type": "reduce_on_plateau",
              "factor": 0.5,
              "mode": "max",
              "patience": int(v["LR_PATIENCE"])
        },
      }
    }

    base["model"]["encoder"] = build_encoder(v)

    print("Writing", args[2])
    with open(args[2], 'w') as ofh:
        json.dump(base, ofh, indent = 2)

    return 0

def build_encoder(v):
    d = {"type": v["MODEL"],
             "hidden_size": int(v["HIDDEN_SIZE"]),
            "num_layers": 2,
            "dropout": 0.5,
            "bidirectional": True,
            "input_size": 178,
         }
    if v["MODEL"] != "lstm":
        d["use_tanh"] = 1
        d["use_relu"] = 0
        d["use_selu"] = 0
        d["rnn_dropout"] = float(v["RNN_DROPOUT"])

    if v["MODEL"] == 'sru':
        d["use_highway"] = bool(v["USE_HIGHWAY"])
        d["use_recurrent_tanh"] = bool(v["USE_RECURRENT_TANH"])
    elif v["MODEL"] == "sopa":
        d["use_highway"] = bool(v["USE_HIGHWAY"])
        d["coef"] = float(v["COEF"])


    return d

if __name__ == '__main__':
    sys.exit(main(sys.argv))
