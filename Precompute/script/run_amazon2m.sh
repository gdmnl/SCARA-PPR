DATASTR=amazon2m
ALGOSTR=featpush
SEED=8
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -feats feats_normt.npy \
        -alpha 0.2 -epsilon 4 -thread_num 1 \
        -seed ${SEED} > ${SAVEDIR}/out_${SEED}.txt
