# scheduling-example
Examples showcasing cases where XLA's scheduler does not work perfectly OOTB

This example runs on 4 GPUs on a single node.
To run, `bash run.sh <unique_name_for_profile>`

This creates an output where the communication kernels are grouped together at the start of the profile.


# expectations
Screenshot of profile generated from this script:
![Screenshot of profile generated from this script](https://github.com/abhinavgoel95/scheduling-example/blob/main/images/bad_overlap.png?raw=true)

Screenshot of expected profile:
![Screenshot of expected profile](https://github.com/abhinavgoel95/scheduling-example/blob/main/images/good_overlap.png?raw=true)
