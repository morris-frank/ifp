# ifp
My Files for my advanced practical at the HCI in Heidelberg

Now with atom.io

## Files

| File                          | Function |
|-------------------------------|----------|
| ConvertDeploy.sh              | Converts the deploy.prototxt from the NIPS snapshots into the used FCN form |
| FCNTest.py                    | Runs all images from a folder through a FCN ad saves maximum activation layer |
| net_surgery.py                | Converts a snapshot from NIPS to a FCN deploy |
| surgery.py                    | Contains function to transplant weights into fcn (todo: merge with net_surgery) creds to shelhammer |
| TestTrain.py                  | Transplant weights into fcn and train it |
