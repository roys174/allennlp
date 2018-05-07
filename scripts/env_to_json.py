#!/usr/bin/env python

import sys
import json


def main(args):
    argc = len(args)

    if argc < 5:
        print("Usage:", args[0], "<env file or '-' to read from ENV> <json base file> <field names (comma separated)> <json output file>")
        return -1

    if args[1] == '-':
        print("Reading args from env variables")
        import os
        v = os.environ
    else:
        print("Reading args from", args[1])
        with open(args[1]) as ifh:
            v = dict([(l.split()[1].split("=")) for l in ifh])

    with open(args[2]) as ifh:
        base = json.load(ifh)

    for encoder in args[3].split(","):
        base["model"][encoder] = build_encoder(v, base["model"][encoder])

    if "trainer" in base:
        if "optimizer" in base["trainer"]:
            base["trainer"]["optimizer"]["lr"] = float(v["LEARNING_RATE"])
        else:
            print("No optimizer found in trainer")
        base["trainer"]["grad_clipping"] = float(v["CLIP_GRAD"])
        base["trainer"]["patience"] = int(v["PATIENCE"])

        if "learning_rate_scheduler" in base["trainer"]:
            base["trainer"]["learning_rate_scheduler"]["patience"] = int(v["LR_PATIENCE"])
        else:
            base["trainer"]["learning_rate_scheduler"] = {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "mode": "max",
                "patience": int(v["LR_PATIENCE"])
            }
    else:
        print("Trainer not in base model")


    if "regularizer" in base and base["regularizer"][0] == "scalar_parameters" \
        and base["regularizer"][0]["type"] == "l2":
        base["regularizer"][0]["alpha": float(v["WEIGHT_DECAY"])]

    print("Writing", args[4])
    with open(args[4], 'w') as ofh:
        json.dump(base, ofh, indent = 2)

    return 0

def build_encoder(v, old_encoder):
    hidden_size = int(v["HIDDEN_SIZE"]) if "HIDDEN_SIZE" in v else old_encoder["hidden_size"]
    d = {"type": v["MODEL"],
             "hidden_size": hidden_size,
            "num_layers": 2,
            "dropout": 0.5,
            "bidirectional": True,
            "input_size": old_encoder["input_size"]
         }
    if v["MODEL"] != "lstm":
        d["use_tanh"] = 1
        d["use_relu"] = 0
        d["use_selu"] = 0
        d["rnn_dropout"] = float(v["RNN_DROPOUT"])

    if v["MODEL"] == 'sru':
        d["use_highway"] = bool(v["USE_HIGHWAY"])
        d["recurrent_tanh"] = bool(v["RECURRENT_TANH"])
    elif v["MODEL"] == "sopa":
        d["use_highway"] = bool(v["USE_HIGHWAY"])
        d["coef"] = float(v["COEF"])


    return d

if __name__ == '__main__':
    sys.exit(main(sys.argv))
