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
  --DONOTCPUS           Switch off using multiple CPUs on the maschine
  --USENCPUS=USENCPUS   Define the number of CPUs to use
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

