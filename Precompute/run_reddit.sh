DATASTR=reddit
ALGOSTR=feat
SEED=0
OUTDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
mkdir -p ${OUTDIR}
OUTFILE=${OUTDIR}/out_${SEED}.txt
build/featpush -algo CLEAN_GRAPH -graph ../data/${DATASTR}/adj.txt -is_undirected no -output_folder ../data/${DATASTR}
build/featpush -algo FEATPUSH -meta ../data/${DATASTR}/attribute.txt -graph_binary ../data/${DATASTR}/graph.bin -query ../data/${DATASTR}/query.txt -feature_file ../data/${DATASTR}/feats_norm.npz -estimation_folder ${OUTDIR} -query_size 232965 -alpha 0.5 -with_idx no -epsilon 64 > $OUTFILE &
tail -f $OUTFILE
