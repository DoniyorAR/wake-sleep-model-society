#!/bin/bash
# All runs of the paper: awake-phase caches first, then every policy on three streams.
# A run whose result file exists is skipped. Set WSMS_ROOT to write somewhere other than ./results.
ROOT="${WSMS_ROOT:-results}"; PY="${PYTHON:-python}"; cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q15=Qwen/Qwen2.5-1.5B-Instruct; Q4=Qwen/Qwen3-4B-Instruct-2507; PHI=microsoft/Phi-4-mini-instruct
O15="$ROOT/qwen2.5-1.5b"; O4="$ROOT/qwen3-4b"; L="$ROOT/logs"; mkdir -p "$O15" "$O4" "$L" "$ROOT/wake"
HOMEO="--refuse 0.15 --anchor 0.3"
flags() { case "$1" in
  naive) echo "--policy naive";; doc_only) echo "--policy doc_only";; oracle) echo "--policy oracle";;
  wsms) echo "--policy gate $HOMEO";;
  wsms_notag) echo "--policy gate --hearsay quarantine $HOMEO";;
  wsms_solo) echo "--policy gate --support self --veto self $HOMEO";;
  wsms_nohomeo) echo "--policy gate";;
  wsms_nosup) echo "--policy gate --support none $HOMEO";;
  wsms_noveto) echo "--policy gate --veto none $HOMEO";;
  wsms_r40) echo "--policy gate --refuse 0.4 --anchor 0.3";; esac; }
cache() { # sleeper helper tag seed
  [ -f "$ROOT/wake/$3_s$4.json" ] || $PY wake.py --sleeper "$1" --helper "$2" --seed "$4" \
    --out "$ROOT/wake/$3_s$4.json" --anchors "$ROOT/wake/anchors_$3.json" > "$L/wake_$3_s$4.log" 2>&1; }
one() { # model outdir tag name seed
  [ -f "$2/$4_s$5.json" ] && return 0
  local cmd="$PY run.py --model $1 --cache $ROOT/wake/$3_s$5.json --anchors $ROOT/wake/anchors_$3.json $(flags "$4") --name $4 --seed $5 --days 30 --mmlu_n 300 --out $2"
  $cmd > "$L/$3_$4_s$5.log" 2>&1 || { sleep 20; $cmd > "$L/$3_$4_s$5.log" 2>&1; }; }
for s in 1 2 3; do cache $Q15 $Q4 q15 $s; cache $Q4 $PHI q4 $s; done
for s in 1 2 3; do
  for n in naive doc_only oracle wsms wsms_notag wsms_solo wsms_nosup wsms_noveto wsms_nohomeo; do one $Q15 "$O15" q15 $n $s; done
  for n in naive doc_only wsms wsms_r40; do one $Q4 "$O4" q4 $n $s; done
done
