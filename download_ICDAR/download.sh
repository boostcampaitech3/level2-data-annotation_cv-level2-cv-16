#!/usr/bin/env bash

for url in $(cat urls.txt | tr -d '\r')
do
    wget $url --no-check-certificate
done