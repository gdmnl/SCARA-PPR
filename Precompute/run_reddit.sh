DATASTR=reddit
ALGOSTR=featpush
SEED=0
DATADIR=../data/${DATASTR}
OUTDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${OUTDIR}
OUTFILE=${OUTDIR}/out_${SEED}.txt
build/featpush -algo CLEAN_GRAPH -graph ${DATADIR}/adj.txt -is_undirected no -output_folder ${DATADIR}
build/featpush -algo FEATPUSH -meta ${DATADIR}/attribute.txt -graph_binary ${DATADIR}/graph.bin -query ${DATADIR}/query.txt -feature_file ${DATADIR}/feats_norm.npz -estimation_folder ${OUTDIR} -with_idx no -query_size 232965 -alpha 0.5 -epsilon 64 > $OUTFILE
