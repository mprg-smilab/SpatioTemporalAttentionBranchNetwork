#!/bin/bash


echo "compress JPEG video frames"


cd /raid/hirakawa/dataset/something-something-v2


### compress frames
# NOTE: sample number is from 1 to 220,847
tar zcf frame-01.tar.gz frame/{1..10000} &
tar zcf frame-02.tar.gz frame/{10001..20000} &
tar zcf frame-03.tar.gz frame/{20001..30000} &
tar zcf frame-04.tar.gz frame/{30001..40000} &
tar zcf frame-05.tar.gz frame/{40001..50000} &
tar zcf frame-06.tar.gz frame/{50001..60000} &
tar zcf frame-07.tar.gz frame/{60001..70000} &
tar zcf frame-08.tar.gz frame/{70001..80000} &
tar zcf frame-09.tar.gz frame/{80001..90000} &
tar zcf frame-10.tar.gz frame/{90001..100000} &

tar zcf frame-11.tar.gz frame/{100001..110000} &
tar zcf frame-12.tar.gz frame/{110001..120000} &
tar zcf frame-13.tar.gz frame/{120001..130000} &
tar zcf frame-14.tar.gz frame/{130001..140000} &
tar zcf frame-15.tar.gz frame/{140001..150000} &
tar zcf frame-16.tar.gz frame/{150001..160000} &
tar zcf frame-17.tar.gz frame/{160001..170000} &
tar zcf frame-18.tar.gz frame/{170001..180000} &
tar zcf frame-19.tar.gz frame/{180001..190000} &
tar zcf frame-20.tar.gz frame/{190001..200000} &

tar zcf frame-21.tar.gz frame/{200001..210000} &
tar zcf frame-22.tar.gz frame/{210001..220000} &
tar zcf frame-23.tar.gz frame/{220001..220847} &
wait


echo "done"

