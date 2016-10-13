#!/bin/bash

echo $1
echo $2

if ! test -f $1 ; then
	echo "No Input prototxt"
	exit
fi

cp $1 $2

## declare an array variable
declare -a arr=("fc6" "fc7_" "fc7" "fc8_output" "fc8")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i-conv"
   sed -e "s/\"${i}\"/\"${i}-conv\"/g" -i.bak $2
done

sed -e "s/InnerProduct/Convolution/g" -i.bak $2
sed -e "s/inner_product_param/convolution_param/g" -i.bak $2
awk '{print; if (NR==257) print "    kernel_size: 6";}' $2 > $2.bak && mv $2.bak $2
awk '{print; if (NR==300) print "    kernel_size: 1";}' $2 > $2.bak && mv $2.bak $2
awk '{print; if (NR==343) print "    kernel_size: 1";}' $2 > $2.bak && mv $2.bak $2