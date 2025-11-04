
data {
  int<lower=0> N; // number of observations
  int Left[N]; // 1/0 did the participant choose the Left lottery?
  
  
  // NOTE: prizes must be ranked from smallest to largest for this to work
  matrix[N,3] prizes; // the 3 lottery prizes
  matrix[N,3] qL; // probabilities for the Left lottery
  matrix[N,3] qR; // probabilities for the Right lottery
  
  vector[2] prior_r; // normal, mean + sd
  vector[2] prior_lambda; // log-normal, transformed mean + sd
}

transformed data {
  
  /* The model only cares about differences in utilities, so we can pre-compute 
  probability differences here
  */
  matrix[N,3] dprob = qL-qR;
  
}


parameters {
  real r;
  real<lower=0> lambda;
}


model {

  // Prior contribution --------------------------------------------------------
  
  target += normal_lpdf(r | prior_r[1],prior_r[2]);
  target += lognormal_lpdf(lambda | prior_lambda[1],prior_lambda[2]);
  
  // Likelihood contribution ---------------------------------------------------
    // lambda * (difference in normalized expected utility)
    vector[N] lDU = lambda*(
      (dprob .* pow(prizes,1.0-r))./
      rep_matrix(
        pow(prizes[,3],1.0-r)-pow(prizes[,1],1.0-r),3
      )
    )*rep_vector(1.0,3);
    
    target += bernoulli_logit_lpmf(Left | lDU);
  
}

