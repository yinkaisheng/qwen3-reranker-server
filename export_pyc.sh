#!/bin/bash
CUR_DIR="$(pwd)"
SCRIPT_DIR="$(dirname "$0")"
SCRIPT_NAME="$(basename "$0")"
BASE_NAME="${SCRIPT_NAME%.*}"
echo "\$0: $0"
echo "CUR_DIR: $CUR_DIR"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "SCRIPT_NAME: $SCRIPT_NAME"
echo "BASE_NAME: $BASE_NAME"
echo "CurrentTime：$(date '+%Y-%m-%d %H:%M:%S')"
echo ""
cd $SCRIPT_DIR

OUT_DIR=/mnt/e/python_apps/docker/qwen3-reranker-server/app_code
DIR_NAME="$(basename "$OUT_DIR")"

rm -f $OUT_DIR/*.py
rm -f $OUT_DIR/*.pyc
py312 gen_git_commit.py
py312 /mnt/e/codes/python/automation/compile_to_pyc.py -f . -o $OUT_DIR
cp version.py $OUT_DIR
rm -f $OUT_DIR/gen_git_commit.pyc
rm -f $OUT_DIR/version.pyc


cd $OUT_DIR/..



tar -czvf $DIR_NAME.tar.gz $DIR_NAME
