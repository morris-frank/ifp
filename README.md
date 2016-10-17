# ifp
My Files for my advanced practical at the HCI in Heidelberg


## Files

| File                          | Function |
|-------------------------------|----------|
| ApplyOverlay.py               | Applies HeatMaps/Segmentations from one folder on the original images from another folder |
| ApplyPatches.py               | Puts the patches from one folder and with their bounding box file onto the images from another folder |
| ApplyPatchesHead.py           | Runs ApplyPatches.py recursively for all folders |
| ConvertDeploy.sh              | Converts the deploy.prototxt from the NIPS snapshots into the used FCN form |
| ExtractPatches.py             | Extracts the patches from the original frames with a given bounding box file |
| ExtractPatchesHead.py         | Runs ExtractPatches.py recursively for all folders |
| FCNTest.py                    | Runs all images from a folder through a FCN ad saves maximum activation layer |
| net_surgery.py                | Converts a snapshot from NIPS to a FCN deploy |
| PrepareSegmentations.py       | Reads through a leveldb of cliques and thresholds all patches and fills the resized segmentations accordingly |
| RunPersonDetectionTest.py     | Runs directory through crf-rnn to detect persons and saves the heatmaps |
| RunPersonDetectionTestHead.py | Runs RunPersonDetectionTest.py recursively for all folders | 
