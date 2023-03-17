DATASTR=ppi
ALGOSTR=featpush
SEED=7
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -feats feats_normt.npy \
        -alpha 0.3 -epsilon 0.5 -thread_num 1 \
        -seed ${SEED} > ${SAVEDIR}/pre_${SEED}.txt
DATADIR=../data/${DATASTR}_train
SAVEDIR=../save/${DATASTR}/${ALGOSTR}_train/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -feats feats_normt.npy \
        -alpha 0.3 -epsilon 0.5 -thread_num 1 \
        -seed ${SEED} > ${SAVEDIR}/pre_${SEED}.txt
# ../Precompute/build/featpush -algo featpush -data_folder ../data/ppi -feats feats_normt.npy -thread_num 1 -seed 7 -alpha 0.3 -epsilon 0.5
# ../Precompute/build/featpush -algo featpca -data_folder ../data/ppi -feats feats_normt.npy -thread_num 1 -seed 7 -alpha 0.3 -epsilon 0.5
