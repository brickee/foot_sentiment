#!/bin/bash
#Replace the variables with your github repo url, repo name, test video name, json named by your UIN
GIT_REPO_URL="https://github.com/brickee/foot_sentiment.git"
REPO="foot_sentiment/part_8"
TYPE="foot_withdrawing"
VIDEO="./marked/8_types_of_actions_trim.mark.mp4"
UIN_JSON="130005135.json"
UIN_PNG="130005135.png"
git clone $GIT_REPO_URL
echo "cd to $REPO"
cd "${PWD}/$REPO"
#Replace this line with commands for running your test python file.
CUDA_VISIBLE_DEVICES=None python foot_sentiment_part8.py --type $TYPE --video $VIDEO
# rename the generated timeLabel.json and figure with your UIN.
echo $UIN_JSON
echo $UIN_PNG
cp timeLabel.json $UIN_JSON
cp timeLabel.png $UIN_PNG
