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
        base["trainer"]["num_epochs"] = int(v["N_EPOCHS"])
        if "optimizer" in base["trainer"]:
            base["trainer"]["optimizer"]["lr"] = float(v["LEARNING_RATE"])
        else:
            print("No optimizer found in trainer")
        base["trainer"]["grad_clipping"] = float(v["CLIP_GRAD"])

        if "patience" not in base["trainer"]:
            base["trainer"]["patience"] = int(v["PATIENCE"])

        if "learning_rate_scheduler" not in base["trainer"]:
            base["trainer"]["learning_rate_scheduler"] = {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "mode": "max",
                "patience": int(v["LR_PATIENCE"])
            }
    else:
        print("Trainer not in base model")


    if "regularizer" in base["model"] and base["model"]["regularizer"][0][0] == "scalar_parameters" \
        and base["model"]["regularizer"][0][1]["type"] == "l2":
        base["model"]["regularizer"][0][1]["alpha"] = float(v["WEIGHT_DECAY"])

    print("Writing", args[4])
    with open(args[4], 'w') as ofh:
        json.dump(base, ofh, indent = 2)

    return 0

def build_encoder(v, old_encoder):
    hidden_size = int(v["HIDDEN_SIZE"]) if "HIDDEN_SIZE" in v else old_encoder["hidden_size"]
    model = v["MODEL"]

    d = {"type": model,
             "hidden_size": hidden_size,
            "num_layers": int(v["DEPTH"]),
            "bidirectional": True,
            "input_size": old_encoder["input_size"]
         }

    if model != 'qrnn':
        d["dropout"] = float(v["DROPOUT"])

    if model != "lstm":
        d["use_tanh"] = 1
        d["use_relu"] = 0
        d["use_selu"] = 0
        d["rnn_dropout"] = float(v["RNN_DROPOUT"])

    if model == 'sru' or model == "sopa":
        d["use_highway"] = bool_str(v["USE_HIGHWAY"])
        if model == 'sru':
            d["recurrent_tanh"] = bool_str(v["RECURRENT_TANH"])
        elif model == "sopa":
            d["coef"] = float(v["COEF"])

    if model == 'qrnn' or model == "sopa":
        d["use_output_gate"] = bool_str(v["USE_OUTPUT_GATE"])
        d["window_size"] = int(v["WINDOW_SIZE"])

    return d

def bool_str(v):
    return v.lower() == 'true'

if __name__ == '__main__':
    sys.exit(main(sys.argv))
