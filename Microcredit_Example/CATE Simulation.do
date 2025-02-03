********************************************************************************
//Author: Niranjan Kumar 
//Project Name: CATE (Conditional Average Treatment Effect)
********************************************************************************

clear all
set seed 12345 
set obs 1000   

********************************************************************************
// TASK -1 Example of CATE with one Covariate 
********************************************************************************


// 1. Generate X ~ N(0,1) or Covariate 
generate double X = rnormal(0,1)

// 2. Random treatment assignment D ~ Bernoulli(0.5)
generate byte D = (runiform() < 0.5)

// 3. Define the treatment effect tau = 4 if X<0; 10 if X>0
generate double tau = cond(X < 0, 4, 10)

// 4. Define the observed outcome Y = D * tau + noise
generate double Y = D * tau + rnormal(0, 0.5)

// 5. Compute CATE for X<0
summarize Y if D==1 & X<0
local mean_1_small = r(mean)

summarize Y if D==0 & X<0
local mean_0_small = r(mean)
local tau_hat_smallX = `mean_1_small' - `mean_0_small'

// 6. Compute CATE for X>0 
summarize Y if D==1 & X>0
local mean_1_big = r(mean)

summarize Y if D==0 & X>0
local mean_0_big = r(mean)

local tau_hat_bigX = `mean_1_big' - `mean_0_big'

// 7. Print CATE estimates
display "tau_hat_smallX = `tau_hat_smallX'" 
display "tau_hat_bigX = `tau_hat_bigX'"




**************************************************************
* Task-2 : Create Another Dataset with Multiple Covariates for CATE 
**************************************************************

**************************************************************
* Step-1 Setup
**************************************************************

clear all
set seed 12345
set obs 1000

**************************************************************
* Step -2. Generate the features X1, X2, ..., X20  ~ N(0,1)
**************************************************************
forvalues j = 1/20 {
    generate double X`j' = rnormal()
}

**************************************************************
* Step-3. Define theta = 1 / (1 + exp(-X3))
**************************************************************
generate double theta = 1 / (1 + exp(-X3))

**************************************************************
* Step-4. Define d ~ Bernoulli( p = 1 / (1 + exp(-(X1 + X2))) )
**************************************************************
generate double p_d = 1 / (1 + exp(-(X1 + X2)))
generate byte d    = (runiform() < p_d)

**************************************************************
* Step-5. Define outcome y:
*  Non-parametric relationship between y and x 
**************************************************************
generate double y = ///
   max(X2 + X3, 0) /// 
   + ((X4 + X5 + X6) / 6) /// 
   + (d * theta) /// 
   + rnormal(0,1)

**************************************************************
* Step-6. Rename X1..X20 -> V1..V20 for consistency
**************************************************************
forvalues j = 1/20 {
    rename X`j' V`j'
}

**************************************************************
* Step-7. (Optional) Order variables to match 
*    i.e. y, theta, d, V1..V20
**************************************************************
order y theta d V1-V20

**************************************************************
* Step-8. Stratified split 50/50 by d 
*    - We create a strata indicator = d
*    - We generate a uniform random number within each strata
*    - Keep half in df_aux, half in df_main
**************************************************************
generate byte strata = d
by strata, sort: generate double u = runiform()
generate byte in_aux = 0
by strata: replace in_aux = 1 if u <= 0.5

**************************************************************
* Step-9. Save df_aux and df_main as separate Stata datasets
*    (Alternatively, you could keep them in memory as separate frames 
*     if using Stata 16+ "frames" commands.)
**************************************************************
preserve
    keep if in_aux
    save df_aux, replace
restore
keep if in_aux == 0
save df_main, replace

**************************************************************
* Step-10. Create a local macro with covariate names
*     "k = ncol(data)-3" in R means k=20
**************************************************************
local k = 20
local covariates ""
forvalues i = 1/`k' {
    local covariates "`covariates' V`i'"
}

display "Covariates: `covariates'"






***********************************************************************
* 1) Clear workspace and import data
***********************************************************************
clear all
global microcredit_path "C:\Users\niranjan.kumar\Downloads\Microcredit_Example"

* Make sure you have the .dta in this folder or specify the full path
use data_rep.dta, clear

***********************************************************************
* 2) Basic variable prep 
***********************************************************************
rename loansamt_total y
rename treatment      d

local covariates "members_resid_bl nadults_resid_bl head_age_bl act_livestock_bl act_business_bl borrowed_total_bl members_resid_d_bl nadults_resid_d_bl head_age_d_bl act_livestock_d_bl act_business_d_bl borrowed_total_d_bl ccm_resp_activ other_resp_activ ccm_resp_activ_d other_resp_activ_d head_educ_1 nmember_age6_16"

display "`covariates'"


keep y d members_resid_bl nadults_resid_bl head_age_bl act_livestock_bl act_business_bl borrowed_total_bl members_resid_d_bl nadults_resid_d_bl head_age_d_bl act_livestock_d_bl act_business_d_bl borrowed_total_d_bl ccm_resp_activ other_resp_activ ccm_resp_activ_d other_resp_activ_d head_educ_1 nmember_age6_16

drop if missing(y, d)

foreach var of local covariates{
	drop if missing(`var')
}


* Drop missing
egen ID = seq(), from(1) to(`=_N')

***********************************************************************
* 3) (Optional) Summaries or checks
***********************************************************************
describe
summarize


shell python $microcredit_path\cate.py 














