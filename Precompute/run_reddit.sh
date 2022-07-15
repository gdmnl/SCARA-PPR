DATASTR=reddit
ALGOSTR=featpush
SEED=0
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
build/featpush -algo CLEAN_GRAPH -graph ${DATADIR}/adj.txt -is_undirected no -output_folder ${DATADIR}
build/featpush -algo FEATPUSH  -with_idx no \
        -meta ${DATADIR}/attribute.txt -graph_binary ${DATADIR}/graph.bin \
        -query ${DATADIR}/query.txt -feature_file ${DATADIR}/feats_norm.npz \
        -estimation_folder ${SAVEDIR} -split_num 1 \
        -alpha 0.5 -epsilon 64 > ${SAVEDIR}/out_${SEED}.txt
