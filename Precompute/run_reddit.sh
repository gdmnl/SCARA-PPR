DATASTR=reddit
ALGOSTR=featpush
SEED=21
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -query query.txt -feats feats_norm.npy \
        -split_num 1 -seed ${SEED} -alpha 0.5 -epsilon 64 > ${SAVEDIR}/out_${SEED}.txt
# ../Precompute/build/featpush -algo featpush -data_folder ../data/reddit -estimation_folder ../save/reddit/featpush/21 -graph adj.txt -query query.txt -feats feats_norm.npy -split_num 1 -seed 21 -alpha 0.5 -epsilon 64
# ../Precompute/build/featpush -algo featreuse -data_folder ../data/reddit -estimation_folder ../save/reddit/featreuse/21 -graph adj.txt -query query.txt -feats feats_norm.npy -split_num 1 -seed 21 -alpha 0.5 -epsilon 64
