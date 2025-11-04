
data {
  int<lower=0> N; // number of observations
  int Left[N]; // 1/0 did the participant choose the Left lottery?
  int nparticipants; // number of participants in experiment
  // row numbers for the start and end of data corresponding to a participant
  // must be sorted
  int idstartend[nparticipants,2]; 
  
  int<lower=1> nmix;
  
  
  
  // NOTE: prizes must be ranked from smallest to largest for this to work
  matrix[N,3] prizes; // the 3 lottery prizes
  matrix[N,3] qL; // probabilities for the Left lottery
  matrix[N,3] qR; // probabilities for the Right lottery
  
  vector[2] prior_r; // normal, mean + sd
  vector[2] prior_lambda; // log-normal, transformed mean + sd
  vector<lower=0>[nmix] prior_mix; // dirichlet
}

transformed data {
  
  /* The model only cares about differences in utilities, so we can pre-compute 
  probability differences here
  */
  matrix[N,3] dprob = qL-qR;
  
}


parameters {
  /* Here we make the identifying assumption that the mixtures are ordered based
  on r. That is, component 1 of the mixture has the smallest r, and component
  nmix has the largest r. This (hopefully) prevents there from being multiple
  posterior modes that make the same predictions, which should make drawing from 
  the posterior easier. 
  */
  
  ordered[nmix] r;
  
  vector<lower=0>[nmix] lambda;
  simplex[nmix] mix;
}


model {

  // Prior contribution --------------------------------------------------------
  
  target += normal_lpdf(r | prior_r[1],prior_r[2]);
  target += lognormal_lpdf(lambda | prior_lambda[1],prior_lambda[2]);
  target += dirichlet_lpdf(mix | prior_mix);
  
  // Likelihood contribution ---------------------------------------------------
  matrix[N,nmix] lDU;
  for (mm in 1:nmix) {
    
    lDU[,mm] = lambda[mm]*(
      (dprob .* pow(prizes,1.0-r[mm]))./
      rep_matrix(
        pow(prizes[,3],1.0-r[mm])-pow(prizes[,1],1.0-r[mm]),3
      )
    )*rep_vector(1.0,3);
    
  }
  for (ii in 1:nparticipants) {
    int start = idstartend[ii,1];
    int end   = idstartend[ii,2];
    
    vector[nmix] like_mix = log(mix);
    for (mm in 1:nmix) {
      
      like_mix[mm] += bernoulli_logit_lpmf(Left[start:end] | lDU[start:end,mm]);
      
    }
    
    target += log_sum_exp(like_mix);
    
    
  }

  
}

