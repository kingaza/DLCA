#!/bin/bash
    
function readDir(){
for fileName in `ls $1`
  do
    if [[ ${fileName:0-7} == '.nii.gz' ]];  # 查找扩展名为 .nii.gz的文件
    then
      python inference.py -j=1 -b=1 --checkpoint="./checkpoint/adam2020/200.ckpt" --input=$1"/"${fileName%%.*} --output="./prediction/"
    fi
  done
}

readDir $1
