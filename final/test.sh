#!/bin/bash
#Replace the variables with your github repo url, repo name, test video name, json named by your UIN

GIT_REPO_URL="https://github.com/brickee/foot_sentiment.git"
REPO="foot_sentiment/final"
TYPE="foot_pacing"
VIDEO="./pacing/marked_only_videos/foot_pacing_test_1.markonly.mp4"

git clone $GIT_REPO_URL
echo "cd to $REPO"
cd "${PWD}/$REPO"
#Replace this line with commands for running your test python file.
CUDA_VISIBLE_DEVICES=None python foot_pacing.py --type $TYPE --video $VIDEO

