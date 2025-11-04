library(tidyverse)
library(haven)
library(rstan)
  options(mc.cores = parallel::detectCores())
  rstan_options(auto_write = TRUE)


################################################################################
# Load the data
################################################################################

# Note: The show-up fee for the undergrads was $10
show_up_fee <-10

D<-"data/HS2023.dta" |>
  read_dta() |>
  filter(
    # remove MBA participants
    mba==0 
    # remove earned endowment participants
    # All of these had endowment == 80. see page 151 of HS2023
    & endowment !=80) |>
  data.frame() |>
  select(id,choice,prize1L:prob4R,endowment,female,age) |>
  rowwise() |>
  mutate(
    frame = ifelse(max(prize1L:prize3R)<=0,"Loss",
                   ifelse(min(prize1L:prize3R)>=0,"Gain", "Mixed")
    )
  ) |>
  ungroup() |>
  # add in the endowment
  mutate(
    Left = 1-choice,
    prize1L = prize1L+endowment+show_up_fee,
    prize2L = prize2L+endowment+show_up_fee,
    prize3L = prize3L+endowment+show_up_fee,
    prize4L = prize4L+endowment+show_up_fee,
    prize1R = prize1R+endowment+show_up_fee,
    prize2R = prize2R+endowment+show_up_fee,
    prize3R = prize3R+endowment+show_up_fee,
    prize4R = prize4R+endowment+show_up_fee
  ) |>
  # there were never four possible prizes
  select(-contains("4")) |>
  rowwise() |>
  mutate(
    prizerangeLow = min(c(prize1L,prize2R,prize3L)),
    prizerangeHigh = max(c(prize1L,prize2R,prize3L))
  ) |>
  ungroup() |>
  filter(!is.na(Left)) |>
  # PROBLEM: sometimes when a prize isn't used in one of the lotteries, the prize is coded differently between Left and Right
  mutate(
    problemPrize3 = prize3L!=prize3R
  ) |>
  mutate(
    prize3L = ifelse(prob3L<1e-6,prize3R,prize3L),
    prize3R = ifelse(prob3R<1e-6,prize3L,prize3R),
    # That seems to have fixed the problem
    problemPrize3 = prize3L!=prize3R,
  ) |>
  # now we have to get the prizes ordered from lowest to highest
  mutate(
    prize1 = ifelse(prize1L<prize3L,prize1L,prize3L),
    prize2 = prize2L,
    prize3 = ifelse(prize1L<prize3L,prize3L,prize1L),
    
    qL1 = ifelse(prize1L<prize3L,prob1L,prob3L),
    qL2 = prob2L,
    qL3 = ifelse(prize1L<prize3L,prob3L,prob1L),
    
    qR1 = ifelse(prize1L<prize3L,prob1R,prob3R),
    qR2 = prob2R,
    qR3 = ifelse(prize1L<prize3L,prob3R,prob1R)
  ) |>
  select(
    id,Left,prize1:prize3,qL1,qL3,qR1,qR3,frame,female,age
  ) |>
  mutate(
    context = paste(prize1,prize2,prize3,frame) ,
    slope = ((qL3-qR3)/(qL1-qR1)) |> round(3)
  ) |>
  # Problem: there are gaps in the id becuase we have dropped some observations
  rename(
    id_OG = id
  ) |>
  mutate(
    id = paste("-",id_OG/10000,"-") |> as.factor() |> as.numeric()
  ) |>
  arrange(id)

idstartend<- D|>
  ungroup() |>
  mutate(rownum = 1:n()) |>
  group_by(id) |>
  summarize(
    start = min(rownum),
    end = max(rownum)
  ) 

D |> 
  saveRDS("data/HS2023cleaned.rds")



################################################################################
# Estimating hierarchical model
################################################################################

file<-"fits/fit_hierarchical.rds" 

if (!file.exists(file)) {
  
  model<-"code/EUT_hierarchical.stan" |>
    stan_model()
  
  
  dStan<-list(
    N = dim(D)[1],
    Left = D$Left,
    nparticipants = D$id |> unique() |> length(),
    id = D$id,
    
    prizes = cbind(D$prize1,D$prize2,D$prize3),
    qL = cbind(D$qL1,1-D$qL1-D$qL3,D$qL3),
    qR = cbind(D$qR1,1-D$qR1-D$qR3,D$qR3),
    
    prior_MU = list(c(0.27,0.36),c(log(30),0.5)),
    prior_TAU = c(1,1),
    prior_OMEGA = 4
  )
  
  Fit<-model |>
    sampling(data=dStan,seed=42,
             pars = c("z"),include=FALSE,
             # No errors except low ESS with the default options
             iter=10000
             )
  
  Fit |>
    saveRDS(file)
  
} else {
  print(
    paste("skipping",file,"- already done")
  )
}


################################################################################
# Estimating one model for each participant
################################################################################


file<-"fits/summary_individual.rds"

if (!file.exists(file)) {
  
  model<-"code/EUT_pooled.stan" |>
    stan_model()
  
  FitSummary<-tibble()
  
  for (ii in (D$id |> unique())) {
    
    d<-D|> filter(id==ii)
    dStan<-list(
      N = dim(d)[1],
      Left = d$Left,
      
      prizes = cbind(d$prize1,d$prize2,d$prize3),
      qL = cbind(d$qL1,1-d$qL1-d$qL3,d$qL3),
      qR = cbind(d$qR1,1-d$qR1-d$qR3,d$qR3),
      
      prior_r = c(0.27,0.36),
      prior_lambda = c(log(30),0.5)
    )
    
    Fit<-model |>
      sampling(data=dStan,seed=42,
               refresh=1000,
               # This runs really fast, so the overhead of between-chain parallelization
               # isn't worth it
               cores=1)
    
    FitSum<-summary(Fit)$summary |>
      data.frame() |>
      mutate(id=ii,id_OG = d$id_OG[1])
    
    FitSum$par<-rownames(FitSum)
    
    FitSummary<-rbind(FitSummary,FitSum)
    
    FitSummary |> 
      saveRDS(file)
    
    
  }
} else {
  print(
    paste("skipping",file,"- already done")
  )
}


################################################################################
# Participant characteristic estimation
################################################################################

file<-"fits/fit_characteristics.rds"

if (!file.exists(file)) {
  
  model<-"code/EUT_characteristics.stan" |>
    stan_model()
  
  d<-D |>
    mutate(age = age |> as.numeric()) |>
    filter(!is.na(age)) 
  
  X<-cbind(rep(1,dim(d)[1]),d$female,d$age)
  
  dStan<-list(
    N = dim(d)[1],
    Left = d$Left,
    
    nX = dim(X)[2],
    X = X,
    
    prizes = cbind(d$prize1,d$prize2,d$prize3),
    qL = cbind(d$qL1,1-d$qL1-d$qL3,d$qL3),
    qR = cbind(d$qR1,1-d$qR1-d$qR3,d$qR3),
    
    prior_beta_r = list(c(0.27,0.36),
                        c(0,0.1),
                        c(0,0.1)
                        ),
    prior_beta_lambda = list(c(log(30),0.5),
                             c(0,0.1),
                             c(0,0.1)
                             )
  )
  
  Fit<-model |>
    sampling(data=dStan,seed=42)
  
  Fit |>
    saveRDS(file)
  
} else {
  print(
    paste("skipping",file,"- already done")
  )
}


################################################################################
# Finite mixture estimation
################################################################################

NMIX<-c(2,3)

for (nmix in NMIX) {
  
  file <- paste0("fits/fit_mix_",nmix,".rds" )
  
  if (!file.exists(file)) {
    
    print(paste("Estimating finite mixture model with",nmix,"components"))
    
    model<-"code/EUT_mixture.stan" |>
      stan_model()
    
    dStan<-list(
      N = dim(D)[1],
      Left = D$Left,
      nparticipants = D$id |> unique() |> length(),
      
      idstartend = idstartend[,c("start","end")],
      
      nmix = nmix,
      
      prizes = cbind(D$prize1,D$prize2,D$prize3),
      qL = cbind(D$qL1,1-D$qL1-D$qL3,D$qL3),
      qR = cbind(D$qR1,1-D$qR1-D$qR3,D$qR3),
      
      prior_r = c(0.27,0.36),
      prior_lambda = c(log(30),0.5),
      prior_mix = rep(1,nmix)
    )
    
    Fit<-model |>
      sampling(data=dStan,seed=42
               )
    
    Fit |>
      saveRDS(file)
    
  } else {
    print(
      paste("skipping",file,"- already done")
    )
  }
  
  
}



################################################################################
# Pooled estimation
################################################################################

file<-"fits/fit_pooled.rds" 

if (!file.exists(file)) {

  model<-"code/EUT_pooled.stan" |>
    stan_model()
  
  dStan<-list(
    N = dim(D)[1],
    Left = D$Left,
    
    prizes = cbind(D$prize1,D$prize2,D$prize3),
    qL = cbind(D$qL1,1-D$qL1-D$qL3,D$qL3),
    qR = cbind(D$qR1,1-D$qR1-D$qR3,D$qR3),
    
    prior_r = c(0.27,0.36),
    prior_lambda = c(log(30),0.5)
  )
  
  Fit<-model |>
    sampling(data=dStan,seed=42)
  
  Fit |>
    saveRDS(file)
  
} else {
  print(
    paste("skipping",file,"- already done")
  )
}