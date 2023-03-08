DATASTR=amazon2m
ALGOSTR=featpush
SEED=7
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -feats feats_normt.npy \
        -alpha 0.2 -epsilon 4 -thread_num 1 \
        -seed ${SEED} > ${SAVEDIR}/pre_${SEED}.txt
DATADIR=../data/${DATASTR}_train
SAVEDIR=../save/${DATASTR}/${ALGOSTR}_train/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -feats feats_normt.npy \
        -alpha 0.2 -epsilon 4 -thread_num 1 \
        -seed ${SEED} > ${SAVEDIR}/pre_${SEED}.txt
# ../Precompute/build/featpush -algo featpush -data_folder ../data/amazon2m -estimation_folder ../save/amazon2m/featpush/7 -feats feats_normt.npy -thread_num 1 -seed 7 -alpha 0.2 -epsilon 4
