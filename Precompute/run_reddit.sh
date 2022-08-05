DATASTR=reddit
ALGOSTR=featpush
SEED=21
DATADIR=../data/${DATASTR}
SAVEDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${SAVEDIR}
# ../Precompute/build/featpush -algo clean_graph -is_undirected no \
#         -graph ${DATADIR}/adj.txt -output_folder ${DATADIR}
../Precompute/build/featpush -algo ${ALGOSTR} \
        -meta ${DATADIR}/attribute.txt -graph_binary ${DATADIR}/graph.bin \
        -query ${DATADIR}/query.txt -feature_file ${DATADIR}/feats_norm.npy \
        -estimation_folder ${SAVEDIR} -split_num 10 -seed ${SEED} \
        -alpha 0.5 -epsilon 64 > ${SAVEDIR}/out_${SEED}.txt
