#!/usr/bin/env bash

listname=${1}
name=${2}
#config="/afs/cern.ch/work/a/ascarpel/private/FeatureLabelling/config.json"

copydir="/eos/user/w/wa105off/CNN/${name}/root/"
#listname="filelist.txt"
echo ${listname}
echo ${copydir}

if [  -f ${listname} ]; then
  echo "${listname} will be erased and created again"
  rm ${listname}
  touch ${listname}
fi

for i in {0..5000} #enter here the range of run numbers you want to save to the OutputList, 633..1199
do

  filename="${name}_hist_${i}.root"

  if [  -f $copydir"/"$filename ]; then
    echo "add file file ${copydir}/${filename} to ${listname}"
    echo ${copydir}/${filename}  >> ${listname}
    #echo -i ${copydir}/${filename} -c ${config} >> ${listname}
  fi

done
