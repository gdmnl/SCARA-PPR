DATASTR=paper
ALGOSTR=featpush
SEED=2
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -feats feats_normt.npy \
        -alpha 0.5 -epsilon 64 -thread_num 32 \
        -seed ${SEED} > ${SAVEDIR}/pre_${SEED}.txt
# ../Precompute/build/featpush -algo featpush -data_folder ../data/paper -feats feats_normt.npy -thread_num 14 -seed 7 -alpha 0.5 -epsilon 64
