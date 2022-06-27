#!/bin/bash



echo "compress mp4 videos"

cd /raid/hirakawa/dataset/something-something-v2

tar zcf video-01.tar.gz video/{1..20000}.mp4 &
tar zcf video-02.tar.gz video/{20001..40000}.mp4 &
tar zcf video-03.tar.gz video/{40001..60000}.mp4 &
tar zcf video-04.tar.gz video/{60001..80000}.mp4 &
tar zcf video-05.tar.gz video/{80001..100000}.mp4 &

tar zcf video-06.tar.gz video/{100001..120000}.mp4 &
tar zcf video-07.tar.gz video/{120001..140000}.mp4 &
tar zcf video-08.tar.gz video/{140001..160000}.mp4 &
tar zcf video-09.tar.gz video/{160001..180000}.mp4 &
tar zcf video-10.tar.gz video/{180001..200000}.mp4 &

tar zcf video-11.tar.gz video/{200001..220847}.mp4 &

wait

echo "done"



