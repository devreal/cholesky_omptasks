#!/bin/bash

last_prgenv=$(module list 2>&1 | grep PrgEnv | cut -d \- -f 2 | cut -d \/ -f 1)


for target in gnu intel cray
do
  module swap PrgEnv-${last_prgenv} PrgEnv-${target}
  module load numlib/intel/mkl/2018.1
  TARGET=${target}_cray make -C singlecom-deps clean all
  last_prgenv=$target
done

module swap PrgEnv-${last_prgenv} PrgEnv-gnu
module load compiler/ompss/17.12-gnu-7.2.0
TARGET=ompss_cray make -C singlecom-deps clean all
TARGET=ompss_cray make -C fine-deps clean all
TARGET=ompss_cray make -C perrank-deps clean all
