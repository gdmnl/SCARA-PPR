DATASTR=reddit
ALGOSTR=featpush
DATADIR=../data/${DATASTR}
for SEED in 0 1 2
do
    SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
    mkdir -p ${SAVEDIR}
    ../Precompute/build/featpush -algo ${ALGOSTR} \
            -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
            -graph adj.txt -feats feats_normt.npy \
            -alpha 0.5 -epsilon 64 -thread_num 32 \
            -seed ${SEED} > ${SAVEDIR}/pre_${SEED}.txt
done
# ../Precompute/build/featpush -algo featpush -data_folder ../data/reddit -feats feats_normt.npy -thread_num 1 -seed 7 -alpha 0.5 -epsilon 64
# ../Precompute/build/featpush -algo featpca -data_folder ../data/reddit -feats feats_normt.npy -thread_num 1 -seed 7 -alpha 0.5 -epsilon 64
