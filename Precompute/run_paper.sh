DATASTR=paper
ALGOSTR=featpush
SEED=21
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -query query.txt -feats feats_norm.npy \
        -alpha 0.5 -epsilon 64 -split_num 10 -thread_num 1 \
        -seed ${SEED} > ${SAVEDIR}/out_${SEED}.txt
# ../Precompute/build/featpush -algo featpush -data_folder ../data/paper -estimation_folder ../save/paper/featpush/21 -graph adj.txt -query query.txt -feats feats_norm.npy -split_num 1 -thread_num 14 -seed 21 -alpha 0.5 -epsilon 64
