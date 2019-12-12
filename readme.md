## For kinetics and species concentration see code and Species.pdf

To run the model you will need `super_batch.sh`, `bt_serial.sh`, `cu.sh` and `Ca_buffer_GHK.py`.
* make sure that all these files are in a directory named `STEPS`.
* run `super_batch.sh` and wait for the jobs to finish.
* if you need to change VDCC numbers to simulate, go to `super_batch.sh` and change the relevant values.
* in `bt_serial.sh` you can set other simulation parameters (e.g. number of jobs etc.).
* after the jobs are complete, run `cu.sh`. It will clean up unnecessary dumps, and tar all the needed pickle files containing the output data.
* download the dump.tar.gz to you local machine, unpack and plot the results using `Ca_Buffer_GHK.ipynb`

