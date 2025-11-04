
data {
  int<lower=0> N; // number of observations
  int Left[N]; // 1/0 did the participant choose the Left lottery?
  
  int<lower=1> nX; // number of characteristics used 
  matrix[N,nX] X; // characteristics
  
  
  // NOTE: prizes must be ranked from smallest to largest for this to work
  matrix[N,3] prizes; // the 3 lottery prizes
  matrix[N,3] qL; // probabilities for the Left lottery
  matrix[N,3] qR; // probabilities for the Right lottery
  
  vector[2] prior_beta_r[nX]; // normal, mean + sd
  vector[2] prior_beta_lambda[nX]; // log-normal, transformed mean + sd
}

transformed data {
  
  /* The model only cares about differences in utilities, so we can pre-compute 
  probability differences here
  */
  matrix[N,3] dprob = qL-qR;
  
}


parameters {
  vector[nX] beta_r;
  vector[nX] beta_lambda;
}


model {

  // Prior contribution --------------------------------------------------------
  
  for (xx in 1:nX) {
    target += normal_lpdf(beta_r[xx] | prior_beta_r[xx][1],prior_beta_r[xx][2]);
    target += normal_lpdf(beta_lambda[xx] | prior_beta_lambda[xx][1],prior_beta_lambda[xx][2]);
  }
  
  vector[N] r = X*beta_r;
  vector[N] lambda = exp(X*beta_lambda);
  
  // Likelihood contribution ---------------------------------------------------
    // lambda * (difference in normalized expected utility)
    vector[N] lDU = lambda.*((
      (dprob .* pow(prizes,1.0-rep_matrix(r,3)))./
      rep_matrix(
        pow(prizes[,3],1.0-r)-pow(prizes[,1],1.0-r),3
      )
    )*rep_vector(1.0,3)
    );
    
    target += bernoulli_logit_lpmf(Left | lDU);
  
}

