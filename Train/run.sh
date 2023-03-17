DATASTR=amazon2m
ALGOSTR=featpush
for SEED in 0 1 2
do
    OUTDIR=../save/${DATASTR}/${ALGOSTR}/${SEED}
    OUTFILE=${OUTDIR}/out_${SEED}.txt
    python -u run_node.py --seed ${SEED} --config ./config/${DATASTR}.json --dev ${1:--1} > ${OUTFILE} &
    echo $! && wait
done
