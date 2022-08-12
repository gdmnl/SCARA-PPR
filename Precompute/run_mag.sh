DATASTR=mag
ALGOSTR=featpush
SEED=21
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -query query.txt -feats feats_norm.npy \
        -split_num 10 -seed ${SEED} -alpha 0.2 -epsilon 16 > ${SAVEDIR}/out_${SEED}.txt
