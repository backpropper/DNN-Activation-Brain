#!/bin/bash

model=$1
kaldipath=$2/kaldi
featpath=$kaldipath/src/featbin
nnetpath=$kaldipath/src/nnetbin

$featpath/compute-mfcc-feats --verbose=2 --sample-frequency=16000 --use-energy=false scp:$model/wav.scp ark:- | \
$featpath/copy-feats --compress=true ark:- ark:$model/mfccfeats.ark
$featpath/compute-cmvn-stats ark:$model/mfccfeats.ark ark:$model/cmvn.ark
$featpath/copy-feats ark:$model/mfccfeats.ark ark:- | $featpath/apply-cmvn ark:$model/cmvn.ark ark:- ark:- | \
$featpath/splice-feats --left-context=3 --right-context=3 ark:- ark:- | \
$featpath/transform-feats $model/final.mat ark:- ark:- | \
$nnetpath/vnnet-forward --no-softmax=true --prior-scale=1.0 --feature-transform=$model/final.feature_transform --class-frame-counts=$model/ali_train_pdf.counts --use-gpu="yes" $model/final.nnet ark:- ark,t:activations
