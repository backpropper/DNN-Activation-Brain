#!/bin/bash

model=$1
kaldipath=$2/kaldi-trunk
featpath=$kaldipath/src/featbin
nnetpath=$kaldipath/src/nnetbin

$featpath/compute-mfcc-feats --verbose=2 --sample-frequency=44100 --use-energy=false scp:$model/wav.scp ark:- | \
$featpath/copy-feats --compress=true ark:- ark,scp:$model/mfccfeats.ark,$model/mfccfeats.scp
$featpath/compute-cmvn-stats scp:$model/mfccfeats.scp ark:$model/cmvn.ark
$featpath/copy-feats scp:$model/mfccfeats.scp ark:- | $featpath/apply-cmvn ark:$model/cmvn.ark ark:- ark:- | \
$featpath/splice-feats --left-context=3 --right-context=3 ark:- ark:- | \
$featpath/transform-feats $model/final.mat ark:- ark:- | \
$nnetpath/vnnet-forward --no-softmax=true --prior-scale=1.0 --feature-transform=$model/final.feature_transform --class-frame-counts=$model/ali_train_pdf.counts --use-gpu="yes" $model/final.nnet ark:- ark,t:activations
