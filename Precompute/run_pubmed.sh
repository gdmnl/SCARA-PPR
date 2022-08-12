DATASTR=pubmed
ALGOSTR=featreuse
SEED=0
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -query query.txt -feats feats_norm.npy \
        -split_num 1 -seed ${SEED} -alpha 0.1 -epsilon 2 > ${SAVEDIR}/out_${SEED}.txt
