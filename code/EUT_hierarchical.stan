
data {
  int<lower=0> N; // number of observations
  int Left[N]; // 1/0 did the participant choose the Left lottery?
  int nparticipants; // number of participants in the experiment
  int id[N]; // id of participant 
  
  
  // NOTE: prizes must be ranked from smallest to largest for this to work
  matrix[N,3] prizes; // the 3 lottery prizes
  matrix[N,3] qL; // probabilities for the Left lottery
  matrix[N,3] qR; // probabilities for the Right lottery
  
  
  vector[2] prior_MU[2];
  vector<lower=0>[2] prior_TAU;
  real prior_OMEGA;
  
  
}

transformed data {
  
  /* The model only cares about differences in utilities, so we can pre-compute 
  probability differences here
  */
  matrix[N,3] dprob = qL-qR;
  
}


parameters {
  
  vector[2] MU;
  vector<lower=0>[2] TAU;
  cholesky_factor_corr[2] L_OMEGA;
  
  matrix[2,nparticipants] z;
  
}

transformed parameters {
  
  vector[nparticipants] r;
  vector[nparticipants] lambda;
  
  {
    matrix[2,nparticipants] theta = rep_matrix(MU,nparticipants)+diag_pre_multiply(TAU,L_OMEGA)*z;
    
    r = theta[1,]';
    lambda = exp(theta[2,]');
  }
}


model {

  
  // Hierarchcial structure ----------------------------------------------------
  target+= std_normal_lpdf(to_vector(z));
  
  // Prior contribution --------------------------------------------------------
  for (pp in 1:2) {
    target += normal_lpdf(MU[pp] | prior_MU[pp][1],prior_MU[pp][2]);
    target += cauchy_lpdf(TAU[pp] | 0.0, prior_TAU[pp]);
  }
  target += lkj_corr_cholesky_lpdf(L_OMEGA | prior_OMEGA);
  
  
  // Likelihood contribution ---------------------------------------------------
    // lambda * (difference in normalized expected utility)
    vector[N] lDU = lambda[id].*((
      (dprob .* pow(prizes,1.0-rep_matrix(r[id],3)))./
      rep_matrix(
        pow(prizes[,3],1.0-r[id])-pow(prizes[,1],1.0-r[id])
        ,3
      )
    )*rep_vector(1.0,3)
    );
    
    target += bernoulli_logit_lpmf(Left | lDU);
  
}

generated quantities{
  matrix[2,2] OMEGA = L_OMEGA*L_OMEGA';
  
}

