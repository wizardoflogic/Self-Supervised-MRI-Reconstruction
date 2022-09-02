# Note: The download period for the dataset will be expired soon - you might need to request fastMRI dataset on your own

mkdir data 
cd data

curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_singlecoil_train.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=2nsh3yC9iliT98oZjGHQ3uouh3w%3D&Expires=1624499945" --output knee_singlecoil_train.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_singlecoil_val.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=lqjhUqOU%2BLXbditX13FbkBtc02w%3D&Expires=1624499945" --output knee_singlecoil_val.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_singlecoil_test_v2.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=kdVCzqWPPYOtwo4d%2BvgKBSUdcOg%3D&Expires=1624499945" --output knee_singlecoil_test_v2.tar.gz

tar -xvf knee_singlecoil_train.tar.gz
rm knee_singlecoil_train.tar.gz

tar -xvf knee_singlecoil_val.tar.gz
rm knee_singlecoil_val.tar.gz

tar -xvf knee_singlecoil_test_v2.tar.gz
rm knee_singlecoil_test_v2.tar.gz

mv singlecoil_test_v2 singlecoil_test
