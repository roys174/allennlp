#!/usr/bin/env bash

cp ~/code/soft_patterns_generation/multilayer_sopa/cuda/* ~/code/allennlp_ner/allennlp/allennlp/modules/multilayer_sopa/cuda/
cp ~/code/soft_patterns_generation/multilayer_sopa/{qrnn,sru*,sopa*}.py ~/code/allennlp_ner/allennlp/allennlp/modules/multilayer_sopa/

for f in ~/code/allennlp_ner/allennlp/allennlp/modules/multilayer_sopa/{sopa,sru,qrnn}.*py; do
    cat $f | sed 's/from multilayer_sopa/from allennlp.modules.multilayer_sopa/' > a
    n=$(grep -n 'if return_hidden:' $f | awk '{print $1}' | tr -d ':')
    sed -i '' ${n}'i\
    \        prevx = pack_padded_sequence(prevx, lengths, batch_first=True)\
    ' a

    n=$(grep -n 'def forward(self, input, init=None, return_hidden=True):' $f | awk '{print $1}' | tr -d ':')
    let n++
    sed -i '' ${n}'i\
    \        input, lengths = pad_packed_sequence(input, batch_first=True)
    ' a

    n=$(grep -n 'from allennlp.modules.multilayer_sopa' a | tail -n 1  |tr ':' ' ' | awk '{print $1}')
    let n++
    sed -i '' ${n}'i\
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    ' a
    mv -f a $f
done

for f in ~/code/allennlp_ner/allennlp/allennlp/modules/multilayer_sopa/{sopa,sru}_gpu.py; do
    cat $f | sed 's/from multilayer_sopa/from allennlp.modules.multilayer_sopa/' > a
    mv -f a $f
done