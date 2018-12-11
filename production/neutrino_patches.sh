#!/bin/bash

# choose view
view=0

#env. variable for eos
EOS_MGM_URL=root://eosuser.cern.ch

# get input
export InputFileEOS=$1

# EOS input file size:
echo ""
echo "InputFileEOS: "$InputFileEOS
echo "InputFileEOSSize: "$(stat -c%s "$InputFileEOS")
echo ""


# isolate run number and subrun number from input file
tmp=${InputFileEOS%.*}
sffx=${tmp#*"_"}
Number=${sffx#*"_"}

echo "File number: "$Number
InputFileLocal="hist_${Number}.root"
echo "InputFileLocal: "$InputFileLocal

# copy input file to scratch disk of worker node
CounterCopyFromEOS=1

while ( [ ! -f $InputFileLocal ] || [ $(stat -c%s "$InputFileLocal") != $(stat -c%s "$InputFileEOS") ] ) && [ $CounterCopyFromEOS -lt 100 ]
do
  echo "Copying input file from EOS, attempt #$CounterCopyFromEOS"
  cp $InputFileEOS $InputFileLocal
  echo "InputFileLocalSize: "$(stat -c%s "$InputFileLocal")
  let CounterCopyFromEOS=CounterCopyFromEOS+1

  if [ $(stat -c%s "$InputFileLocal") != $(stat -c%s "$InputFileEOS") ]
  then
    rm -f $InputFileLocal
  fi
done

# output
export OutputFileLocal="db_view_${view}_${Number}.tar.gz"
export OutputPathEOS="ascarpel@tlab-gpu-pdune-01:/data/ascarpel/images/particlegun/"
export OutputFileEOS=$OutputPathEOS"/"$OutputFileLocal
mkdir -p $OutputPathEOS
rm -f $OutputFileEOS

# run script:
# make a fodler where to store all the .png images
# compress the folder
# copy compress folder to eos
export ZipFolder="dbimages${view}_${Number}"
mkdir $ZipFolder
python /afs/cern.ch/work/a/ascarpel/private/FeatureLabelling/prepare_patches_em-trk-michel-none.py -i $InputFileLocal -o $ZipFolder -v $view
tar -zcvf $OutputFileLocal $ZipFolder

scp $OutputFileLocal $OutputPathEOS

# copying the npy output file to eos sometimes fails. Try max 100 times. Also, sometimes the file is on eos, but is empty (0 bytes) or has 731 bytes. In that case, copy again.
#CounterCopyToEOS=1
#while ( [ ! -f $OutputFileEOS ] || [ $(stat -c%s "$OutputFileEOS") != $(stat -c%s "$OutputFileLocal") ] ) && [ $CounterCopyToEOS -lt 10 ]
#do
 # echo "Copying text output file to EOS, attempt #$CounterCopyToEOS"
 # scp $OutputFileLocal $OutputPathEOS
 # echo "File size on EOS: "$(stat -c%s "$OutputFileLocal")
 # let CounterCopyToEOS=CounterCopyToEOS+1

 # if [ $(stat -c%s "$OutputFileEOS") != $(stat -c%s "$OutputFileLocal") ]
 # then
 #  rm -f $OutputFileEOS
 # fi

  # Untar file on eos
  #tar -xvf $OutputFileEOS -C $OutputPathEOS
  #rm $OutputFileEOS

#done

#tar -xvf $OutputFileEOS -C $OutputPathEOS
#rm $OutputFileEOS

rm -f $InputFileLocal
rm -rf $ZipFolder
rm -f $OutputFileLocal
