# HeterogeneityForThePractitioner
Code for "Adding parameter heterogeneity to structural models: some guidance for the practitioner" by James R. Bland

This repository replicates all estimations in the paper. 

The "data" folder contains instructions for downloading the data from Glenn Harrison's data archive. Once the data are in this folder, set your working directory to the directory that this README file is in. Then run `code/RunEstimation.R`. This will run all of the estimations, but not produce any tables or figures. The tables and figures are produced by compiling `Heterogeneity.Rmd`, which completely reproduces the manuscript as it was submitted to the *Journal of Behavioral and Experimental Economics* on 2026-09-28.  

You will need the following R libraries to run the code and compile the manuscript:
* `tidyverse`
* `haven`
* `rstan`
* `kableExtra`
* `viridis`

Within "GSU.dta", the relevant variables are:

* `id`, a unique id code for each participant
* `prize1L:prize3R`, the prizes of the Left and Right lotteries as displayed to participants
* `prob1L:prob3R`, the probabilities associated with the Left and Right lotteries
* `endowment`, the framed endowment for each lottrey pair (this is added to the prizes so that they are all positive)
* `choice`, and indicator variable. =1 if the Right lottery was chosen, =0 if the Left lottery was chosen
* `female`, and indicator variable. =1 if the participant is female, =0 otherwise
* `age`, participant age in years.
