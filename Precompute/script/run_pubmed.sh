DATASTR=pubmed
ALGOSTR=featpush
SEED=7
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -data_folder ${DATADIR} -estimation_folder ${SAVEDIR} \
        -graph adj.txt -feats feats_normt.npy \
        -alpha 0.1 -epsilon 2 -thread_num 1 \
        -seed ${SEED} > ${SAVEDIR}/pre_${SEED}.txt
# ../Precompute/build/featpush -algo featpush -data_folder ../data/pubmed -graph adj.txt -feats feats_normt.npy -thread_num 1 -seed 0 -alpha 0.1 -epsilon 2
# ../Precompute/build/featpush -algo featpca -data_folder ../data/pubmed -graph adj.txt -feats feats_normt.npy -thread_num 1 -seed 0 -alpha 0.1 -epsilon 2
