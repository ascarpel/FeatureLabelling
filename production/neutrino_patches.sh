#!/bin/bash

#scl enable python27 bash
#source /afs/cern.ch/sw/lcg/external/gcc/4.9/x86_64-slc6-gcc49-opt/setup.sh
#source /afs/cern.ch/sw/lcg/app/releases/ROOT/6.06.06/x86_64-slc6-gcc49-opt/root/bin/thisroot.sh

# choose view
view=1

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
InputFileLocal="neutrino_hist_"$Number".root"

echo "InputFileLocal: "$InputFileLocal

# copy input file to scratch disk of worker node
CounterCopyFromEOS=1

while ( [ ! -f $InputFileLocal ] || [ $(stat -c%s "$InputFileLocal") != $(stat -c%s "$InputFileEOS") ] ) && [ $CounterCopyFromEOS -lt 100 ]
do
  echo "Copying input file from EOS, attempt #$CounterCopyFromEOS"
  cp $InputFileEOS .
  echo "InputFileLocalSize: "$(stat -c%s "$InputFileLocal")
  let CounterCopyFromEOS=CounterCopyFromEOS+1

  if [ $(stat -c%s "$InputFileLocal") != $(stat -c%s "$InputFileEOS") ]
  then
    rm -f $InputFileLocal
  fi
done

# output
export OutputFileLocal="db_view_${view}_${Number}.tar.gz"
export OutputPathEOS="/eos/user/a/ascarpel/CNN/neutrino/images/tar"
export OutputFileEOS=$OutputPathEOS"/"$OutputFileLocal
mkdir -p $OutputPathEOS
rm -f $OutputFileEOS

# run script:
# make a fodler where to store all the .png images
# compress the folder
# copy compress folder to eos
export ZipFolder="dbimages${view}"
mkdir $ZipFolder
python /afs/cern.ch/work/a/ascarpel/private/FeatureLabelling/prepare_patches_em-trk-michel-none.py -i $InputFileLocal -o $ZipFolder -v $view
tar -zcvf $OutputFileLocal $ZipFolder

# copying the npy output file to eos sometimes fails. Try max 100 times. Also, sometimes the file is on eos, but is empty (0 bytes) or has 731 bytes. In that case, copy again.
CounterCopyToEOS=1
while ( [ ! -f $OutputFileEOS ] || [ $(stat -c%s "$OutputFileEOS") != $(stat -c%s "$OutputFileLocal") ] ) && [ $CounterCopyToEOS -lt 10 ]
do
  echo "Copying text output file to EOS, attempt #$CounterCopyToEOS"
  cp $OutputFileLocal $OutputPathEOS
  echo ".npy file size on EOS: "$(stat -c%s "$OutputFileLocal")
  let CounterCopyToEOS=CounterCopyToEOS+1

  if [ $(stat -c%s "$OutputFileEOS") != $(stat -c%s "$OutputFileLocal") ]
  then
    rm -f $OutputFileEOS
  fi
done

rm -f $InputFileLocal
rm -rf $ZipFolder
