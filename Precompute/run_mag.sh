DATASTR=mag
ALGOSTR=featpush
SEED=7
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -feats feats_norm.npy \
        -alpha 0.2 -epsilon 16 -thread_num 14 \
        -seed ${SEED} > ${SAVEDIR}/out_${SEED}.txt
# ../Precompute/build/featpush -algo featpush -data_folder ../data/mag -estimation_folder ../save/mag/featpush/7 -feats feats_norm.npy -thread_num 1 -seed 7 -alpha 0.2 -epsilon 16
