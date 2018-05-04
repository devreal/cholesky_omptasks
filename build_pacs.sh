#!/bin/bash

export PATH=$PATH:$WORK/opt/ompss-17.12-icc/bin

for target in intel
do
  TARGET=${target}_pacs make -k -C singlecom-deps clean all
done

TARGET=ompss_pacs make -k -C singlecom-deps clean all
TARGET=ompss_pacs make -k -C fine-deps clean all
TARGET=ompss_pacs make -k -C perrank-deps clean all
