DATASTR=mag
ALGOSTR=featpush
SEED=21
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -query query.txt -feats feats_norm.npy \
        -alpha 0.2 -epsilon 16 -split_num 1 -thread_num $1 \
        -seed ${SEED} > ${SAVEDIR}/out_${SEED}.txt
