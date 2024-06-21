# RFIWIPER
Radio frequency interference software

Base is first a HDF5 file from SKAMPI to understand some of the issues we encountered for uncalibrated datasets
The first test dataset is EDD_2023-05-19T05_42_23.848010UTC_yWRaJ.hdf5 and is availible via:

	ftp://ftp.mpifr-bonn.mpg.de/outgoing/hrk/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ.hdf5

 more to come in the future.


```
python CHECK_SURVEY_SCANS.py -h

Usage: CHECK_SURVEY_SCANS.py [options]

Options:
  -h, --help            show this help message and exit
  --DATA_FILE=DATAFILE  DATA - HDF5 file of the Prototyp
  --USEDATA=USEDATA     use data noise diode off and on "['ND0','ND1']",
                        default is ['ND0']
  --DONOTFLAG           Do not flag the data.
  --PROCESSING_TYPE=FLAGPROCESSING
                        setting how accurate/much time the flagging proceed.
                        FAST, SEMIFAST, SLOW, default is SEMIFAST
  --DO_FG_TIME_BY_HAND=HAND_TIME_FG
                        use the time index of the waterfall plot e.g.
                        [[0,10],[100,110]]
  --DO_FG_TIME_AUTO_SIGMA=AUTO_TIME_FG_SIGMA
                        automatically determine bad time use threshold.
                        default = 0 is off use e.g. = 5
  --DO_FG_BOUNDARY_SIGMA=BOUND_SIGMA_INPUT
                        if the spectram is maske to much at teh edges
                        increase. [default = 3 sigma]
  --DOPLOT_FINAL_SPEC   Plot the final spectrum after Flagging
  --FINAL_SPEC_YRANGE=FSPEC_YRANGE
                        [ymin,ymax]
  --DOPLOT_FINAL_WATERFALL
                        Plot the final waterfall after Flagging
  --DOPLOT_WITH_INVERTED_MASK
                        Plot the final plots using an inverted mask
  --DOSAVEPLOT          Save the plots as figures
  --EDIT_FLAG           Switch to replace the old with the new mask
  --RESET_FLAG          Switch to clear all mask
  --DOSAVEMASK=SAVEMASK
                        Save the mask into numpy npz file.
  --DOSAVEFINALSPECTRUM=SAVEFINALSPECTRUM
                        Safe the final 1d spectra as numpy npz file. [works
                        only with --DOPLOT_FINAL_SPEC]
  --DOLOADMASK=LOADMASK
                        Upload the mask.
  --DONOTCPUS           Switch off using multiple CPUs on the machine
  --USENCPUS=USENCPUS   Define the number of CPUs to use
  --HEAT_BACKEND        Use the Heat backend for parallelization and GPU support
  --SILENCE             Switch off all output
  --DO_RFI_REPORT=DO_RFI_REPORT
                        provides info and SPWD plots. Input is number of SPWD
                        [default = -1, use e.g. 8]
  --HELP                Show info on input

```



## Lets have a go on the file

- Just look at the **original dataset without flagging**

```
python CHECK_SURVEY_SCANS.py --DATA_FILE=EDD_2023-05-19T05_42_23.848010UTC_yWRaJ.hdf5 --DONOTFLAG --DOPLOT_FINAL_WATERFALL --DOPLOT_FINAL_SPEC --FINAL_SPEC_YRANGE='[-2E12,2E12]' --DOSAVEPLOT
```

Waterfall Spectrum per polarisation (P0/P1)

![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P0_ND0_WFPLT.png" width=25%>
![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P1_ND0_WFPLT.png" width=25%>

Averaged Spectrum (mean) and the standart derivation as error's in red per polarisation (P0/P1)

![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P0_ND0_SPEC.png" width=25%>
![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P1_ND0_SPEC.png" width=25%>


- Just **flag by hand** some times

```
python CHECK_SURVEY_SCANS.py --DATA_FILE=EDD_2023-05-19T05_42_23.848010UTC_yWRaJ.hdf5 --DONOTFLAG --DOPLOT_FINAL_SPEC --FINAL_SPEC_YRANGE='[-2E12,2E12]' --DOPLOT_FINAL_WATERFALL --DO_FG_TIME_BY_HAND='[[0,40],[1695,1750],[3405,3455],[5114,5162],[6820,6875]]' --DOSAVEPLOT
```

This also provides some output e.g. for the first entry of the --DO_FG_TIME_BY_HAND settings

	- Hand FG in time
                 idx:  [0, 40]
                 timerange:  [['2023-05-19T05:42:44.695'] ['2023-05-19T05:42:50.831']]
                 azimut range:  62.49950015950033 63.723815246821935
                 elevation range:  30.716222703473235 30.716722503502545


Waterfall Spectrum per polarisation (P0/P1)

![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P0_ND0_WFPLT_HFG.png" width=25%>
![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P1_ND0_WFPLT_HFG.png" width=25%>

Averaged Spectrum (mean) and the standart derivation as error's in red per polarisation (P0/P1)

![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P0_ND0_SPEC_HFG.png" width=25%>
![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P1_ND0_SPEC_HFG.png" width=25%>



- Now do the **full flagging**

```

python -W ignore CHECK_SURVEY_SCANS.py --DATA_FILE=EDD_2023-05-19T05_42_23.848010UTC_yWRaJ.hdf5 --DO_FG_TIME_AUTO_SIGMA=5 --DOPLOT_FINAL_WATERFALL --DOPLOT_FINAL_SPEC --DOSAVEPLOT --DOSAVEMASK=FULL_FLAG_MASK --DONOTCPUS

```

Note: that the setting --DONOTCPUS is sometimes faster than using
ncpus (if the number is small < 10). Runs 0.8 sec per spectrum ~ 1.5 hours for 1 polarisation

Waterfall Spectrum per polarisation (P0/P1)

![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P0_ND0_WFPLT_FULLFG.png" width=25%>
![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P1_ND0_WFPLT_FULLFG.png" width=25%>

Averaged Spectrum (mean) and the standart derivation as error's in red per polarisation (P0/P1)

![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P0_ND0_SPEC_FULLFG.png" width=25%>
![]()<img src="Plots/EDD_2023-05-19T05_42_23.848010UTC_yWRaJ_scan_000_P1_ND0_SPEC_FULLFG.png" width=25%>

## How to use the Heat backend
(C. Comito, email c.comito AT fz-juelich.de for questions)

This branch uses the [Heat](https://github.com/helmholtz-analytics/heat) library under the hood for parallel processing and GPU support.

The main difference to the original implementation is that the flagging is performed in batch on all timestamps of an observation at once. 

### Environment setup 

You don't need to install Heat manually. The Heat library (latest version of the development branch) is included as a submodule in this repository. 

However, you need to have the following packages installed for Heat to run:
 - MPI (e.g. OpenMPI)
 - mpi4py
 - torch (PyTorch) tailored to the specific CUDA version of your system

### How to use the Heat backend

In the `RFIWIPER` directory, check out the `heat_backend` branch.

```
git checkout heat_backend
```

This branch contains a submodule pointing to a specific development branch of the Heat library. To initialize the submodule, run

```
git submodule update --init
```

The directory `RFIWIPER/heat` should now contain a clone of the Heat repository. 


To run the Heat backend implementation, simply add the `--HEAT_BACKEND` flag to the command line. For example:

```
python CHECK_SURVEY_SCANS.py --DATA_FILE=EDD_2023-05-19T05_42_23.848010UTC_yWRaJ.hdf5 --DONOTCPUS --PROCESSING_TYPE=FAST --HEAT_BACKEND
````

### Parallelization

The Heat backend uses MPI for parallelization. Do not use the `--USENCPUS` flag with the Heat backend. The number of MPI processes used is determined by the MPI configuration. For example, to run the job on 4 MPI processes, use the following command within your sbatch script:

```
srun -n 4 python CHECK_SURVEY_SCANS.py --DATA_FILE=EDD_2023-05-19T05_42_23.848010UTC_yWRaJ.hdf5 --PROCESSING_TYPE=FAST --HEAT_BACKEND
```
You can use other sbatch options to specify the number of nodes, processes (tasks) per node, etc.

### GPU usage

Important: 
- You need to run the job on a GPU node
- the PyTorch version installed should match the specific CUDA version of your system/container.
- ideally, make sure your MPI is "CUDA-aware" 
- Set the `CUDA_VISIBLE_DEVICES` environment variable. I.e., if you have 2 GPUs per node available, add 
 
```
export CUDA_VISIBLE_DEVICES=0,1
``` 
to your sbatch script to use both GPUs. Heat can distribute the workload across multiple GPUs, and even across multiple nodes if you have a multi-node setup.

On the GLOW cluster, you have 2 GPU nodes with 1 GPU each. To distribute the job e.g. across 2 GPUs on 2 nodes, use the following command (I GUESS BUT I CANNOT TEST IT):

```
srun --nodes=2 --ntasks-per-node=1 --gres=gpu:0 python CHECK_SURVEY_SCANS.py --DATA_FILE=EDD_2023-05-19T05_42_23.848010UTC_yWRaJ.hdf5 --PROCESSING_TYPE=FAST --HEAT_DEVICE=gpu
```

If you set the --HEAT_DEVICE flag, you can omit the --HEAT_BACKEND flag. 
If you use the --HEAT_BACKEND flag without specifying the device, the default is the CPU(s).

### TODOs

Still to do (in order of decreasing impact):
- [ ] Compare Heat backend results with the original implementation (implement correctness tests)
- [ ] Heat results are disconnected from the plotting routines
- [ ] Use ht.load_hdf5 for parallel loading of HDF5 files
- [ ] Support masked arrays in Heat
- [ ] Use Heat operations in checkerstats_ht instead of numpy
- [ ] vectorize 1D interpolation in boundary_range_ht



