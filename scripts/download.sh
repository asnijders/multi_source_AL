#!/bin/bash

cd $HOME/active_learning/resources/data

## snli ; data/snli_1.0
if [[ ! -d snli_1.0 ]]; then
  curl -Lo snli_1.0.zip 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
  unzip snli_1.0.zip
  rm snli_1.0.zip
  rm -r __MACOSX
fi

## mnli ; data/multinli_1.0
if [[ ! -d multinli_1.0 ]]; then
  curl -Lo multinli_1.0.zip 'https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip'
  unzip multinli_1.0.zip
  rm multinli_1.0.zip
  rm -r __MACOSX
fi

## anli ; data/anli_v1.0
if [[ ! -d anli_v1.0 ]]; then
  curl -Lo anli_v1.0.zip 'https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip'
  unzip anli_v1.0.zip
  rm anli_v1.0.zip
  rm -r __MACOSX
fi

## fever ; data/nli_fever
if [[ ! -d nli_fever ]]; then
  curl -Lo nli_fever.zip 'https://www.dropbox.com/s/hylbuaovqwo2zav/nli_fever.zip?dl=0'
  unzip nli_fever.zip
  rm nli_fever.zip
  rm -r __MACOSX
fi
