#!/bin/bash

if [[ $1 == "all" ]]
then
	rm sorted/*
	rm pending/*
	rm buckets/*
elif [[ $1 == "restart" ]]
then
	mv pending/* buckets/.
fi
