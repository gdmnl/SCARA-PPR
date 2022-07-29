DATASTR=pubmed
ALGOSTR=featreuse
SEED=0
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
../Precompute/build/featpush -algo clean_graph -is_undirected no \
        -graph ${DATADIR}/adj.txt -output_folder ${DATADIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -meta ${DATADIR}/attribute.txt -graph_binary ${DATADIR}/graph.bin \
        -query ${DATADIR}/query.txt -feature_file ${DATADIR}/feats_norm.npy \
        -estimation_folder ${SAVEDIR} -split_num 1 -seed ${SEED} \
        -alpha 0.1 -epsilon 2 > ${SAVEDIR}/out_${SEED}.txt
