DATASTR=reddit
ALGOSTR=featreuse
SEED=7
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -feats feats_normt.npy \
        -alpha 0.5 -epsilon 64 -thread_num 1 \
        -seed ${SEED} > ${SAVEDIR}/out_${SEED}.txt
# ../Precompute/build/featpush -algo featpush -data_folder ../data/reddit -estimation_folder ../save/reddit/featpush/7 -feats feats_norm.npy -thread_num 1 -seed 7 -alpha 0.5 -epsilon 64
# ../Precompute/build/featpush -algo featreuse -data_folder ../data/reddit -estimation_folder ../save/reddit/featreuse/7 -feats feats_norm.npy -thread_num 1 -seed 7 -alpha 0.5 -epsilon 64
