#!/bin/bash

################################################################################
# Make sure that all the required jobs are completed
# Make lists with completed jobs and list with failed jobs
################################################################################

export argumentList=$1
export eosfolder="/eos/user/a/ascarpel/CNN/particlegun/images/"
export filelist="/eos/user/a/ascarpel/CNN/particlegun/files.list"

#make file.list ================================================================
if [ ! -f $filelist  ]; then
  touch $filelist
else
  rm $filelist
  touch $filelist
fi

#check output ==================================================================
while read line; do

  tmp=${line%.*}
  sffx=${tmp#*"_"}
  num=${sffx#*"_"}

  export failview0=false
  export failview1=false
  export filenum=0

  echo $num

  #check view 0 ================================================================
  if [ -d ${eosfolder}"dbimages0_${num}"  ]; then

    echo ${eosfolder}"dbimages0_${num}"
    ls -f ${eosfolder}"dbimages0_${num}"/* >> $filelist

  else
      failview0=true
  fi

  #check view 1 ================================================================
  if [ -d ${eosfolder}"dbimages1_${num}"  ]; then

    echo ${eosfolder}"dbimages1_${num}"
    ls -f ${eosfolder}"dbimages1_${num}"/* >> $filelist

  else
    failview1=true
  fi

done < ${argumentList}
