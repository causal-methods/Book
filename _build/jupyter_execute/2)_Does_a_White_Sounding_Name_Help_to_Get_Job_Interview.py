# 2) Does a White-Sounding Name Help to Get a Job Interview?

Vitor Kamada

E-mail: econometrics.methods@gmail.com

Last updated: 8-8-2020

Let's load the dataset from Bertrand & Mullainathan (2004).

import pandas as pd
path = "https://github.com/causal-methods/Data/raw/master/" 
df = pd.read_stata(path + "lakisha_aer.dta")
df.head(4)

Let's restrict the analysis to the variables 'call' and 'race'. 

call: 1 = applicant was called back to interview; and 0 otherwise.

race: w = White, and b = Black.

callback = df.loc[:, ('call', 'race')]
callback

Let's calculate the number of observations (size) and the mean of the variable 'call' broken by race.

We have the same quantity (2435) curriculum vitae (CVs) for Black and White applicants.

Only 6.4% of Blacks received a callback; whereas 9.6% of Whites received a callback. 

Therefore, White applicants are about 50% more likely to receive a callback for interview.

In other words, for each 10 CVs that White applicants send to get 1 job interview, Black applicants need to send 15 CVs to get the same outcome.

import numpy as np
callback.groupby('race').agg([np.size, np.mean])

Somebody might argue that this difference of 3.2% (9.6 - 6.4) does not necessary implies discrimination against Blacks. 

You could argue that White applicants receives more callbacks, because they have more education, experience, skills, and not because of the skin color. 

Specifically, you could argue that White applicant is more likely to receive a callback, because they are more likely to have a college degree (signal of qualifications) than Blacks. 

If you extract a random sample of US population or check data from US Census, you will see that Blacks are less likely to have a college degree than Whites. This is an undisputable fact.

Let's check the proportion of Blacks and Whites with college degree in the dataset from Bertrand & Mullainathan (2004)

Originally, college graduate was coded as 4 in the variable 'education'. 3 = some college. 2 = high school graduate. 1 = some high school. 0 not reported.

Let's create the variable 'college' = 1, if a person completes a college degree; and 0 otherwise.

df['college'] = np.where(df['education'] == 4, 1, 0)

We can see that 72.2% of Black Applicants have a college degree. The proportion of Whites with college degree is very similar 71.6%. 

Why these numbers are not representative of US population and the values closer to each other? 

Because the data is not a random sample extraction from reality.  

college = df.loc[:, ('college', 'race')]
college.groupby('race').agg([np.size, np.mean])

Bertrand & Mullainathan (2004) produced experimental data. They created the CVs. They randomly assigned a Black sounding name (ex: Lakish or Jamal) to half of the CVs and a White sounding name (ex: Emily or Greg) to the other half. 

Randomization of the race via name makes the two categories White and Black equal (very similar) to each other for all observable and non-observable factors.

Let's check this statement for other factors in the CVs. The names of the variables are self-explanatory, and more information can be obtained reading the paper from Bertrand & Mullainathan (2004).

resume = ['college', 'yearsexp', 'volunteer', 'military',
          'email', 'workinschool', 'honors',
          'computerskills', 'specialskills']
both = df.loc[:, resume]
both.head()

Let's use a different code to calculate the mean of the variables for the whole sample (both Whites and Blacks) and broken samples between Blacks and Whites. 

See that the average years of experience (yearsexp) is 7.84 for the whole sample, 7.82 for Blacks, and 7.85 for Whites. 

If you check all variables the mean values for Blacks are very closer to the mean values for Whites. This is the consequence of randomization.

We also calculate the standard deviation (std), a measure of variation around the mean. Note that the standard deviation is pretty much the same between the whole sample and split samples. Like the mean, you don't suppose to see much difference among standard deviations in the case of experimental data.

The standard deviation of the variable years of experience is 5 years. We can state roughly the most part of observations (about 68%) is between 1 std below the mean and 1 std above the mean, that is, between [2.84, 12.84].

black = both[df['race']=='b']
white = both[df['race']=='w']
summary = {'mean_both': both.mean(),   'std_both': both.std(),
           'mean_black': black.mean(), 'std_black': black.std(),
           'mean_white': white.mean(), 'std_white': white.std()}
pd.DataFrame(summary)

Why we care so much about the table above that shows that the average White and average Black applicants are pretty much the same? 

Because the argument that White applicants are more likely to receive a callback due to their higher level of education cannot hold, if both groups White and Black applicants have similar level of education. 

Neither unobserved factors nor unmeasurable factors like motivation, psychological traits, etc., can be used to justify the different rate of callbacks. 

In an experiment, only the treatment variable (race) is exogenously manipulated. Everything else is kept constant, consequently variations in the outcome variable (callbacks) can only be attribute to the variations in the treatment variable (race). 

Therefore, experimental study eliminates all confound factors presented in observational studies. 

Experiment is the gold standard in Hard Science. The most rigorous way to claim causality. 

Surveys, census, observational data direct extracted from reality cannot be used to establish causality. It might useful to capture association, but not causation.

Formally, we can write a causal model below. The 3 lines are equivalent. We can claim that $\beta$ has "causal" interpretation, only if the treatment variables ($T$) was randomized. In the absence of randomization, $\beta$ captures only correlation. 

$$Outcome = Intercept + Slope*Treatment + Error$$

$$y=Intercept+\beta T +\epsilon$$

$$callbacks = Intercept+\beta race+\epsilon$$


Let's estimate the model above, using the ordinary least square (OLS) method.

In the stasmodels library of Python, the intercept is a constant with value of 1.

Let's create the variable 'treatment': 1 = Black applicant, and 0 = White applicant.

The variable 'call' is the outcome variable (y).

df['Intercept'] = 1
df['Treatment'] = np.where(df['race'] =='b', 1, 0)
import statsmodels.api as sm
ols = sm.OLS(df['call'], df[['Intercept', 'Treatment']],
                    missing='drop').fit()

Let's print the results.



print(ols.summary().tables[1])

Now we can write the fitted model as:

$$\widehat{callbacks} = 0.0965-0.032\widehat{Treatment}$$

We already got the coefficients above with the code in the beginning of this section that I will reproduce again:

callback.groupby('race').agg([np.size, np.mean])

See that the value of Intercept is 9.6%. This is the proportion of White applicants that received a callback for interview. 

The coefficient of the treatment variable is 3.2%. The interpretation is that being a Black apllicant "causes" to receive -3.2% ( 6.4% - 9.6%) less callbacks for interview. 

Remember that 3.2% is a big magnitude, as it represents about 50% differential. In practical terms, Black applicants has to send 15 CVs to secure one interview rather than 10 CVs for White applicants.

The coefficient of the treatment variable is also statistically significant at level of significance ($\alpha$ = 1%).

The t-value of -4.115 is the ratio:

$$t = \frac{coefficient}{standard\ error} =\frac{-0.032}{0.008} = -4.115$$
  

The null hypothesis is:

$$H_0: \beta=0$$

The t-value of -4 means that the observed value (-3.2%) is 4 standard deviation below the mean ($\beta=0$). The p-value or probability  of getting this value at chance is virtually 0. Therefore, we reject the null hypothesis that the magnitude of treatment is 0.

What defines an experiment?

The randomization of the treatment variable (T).

It automatically makes the treatment variable (T) independent of other factors:

$$T \perp Other \ Factors$$

In an experiment, the addition of other factors in the regression cannot affect the estimation of the coefficient of the treatment variable ($\beta$). If you see substantial changes in $\beta$, you can infer that you are not working with experimental data.  

Note that in observational studies, you must always control for other factors. Otherwise, you will have the omitted variable bias problem. 

Let's estimate the multiple regression below:

$$y=Intercept+\beta T + Other\ Factors+\epsilon$$

other_factors = ['college', 'yearsexp', 'volunteer', 'military',
          'email', 'workinschool', 'honors',
          'computerskills', 'specialskills']
multiple_reg = sm.OLS(df['call'],
                      df[['Intercept', 'Treatment'] + other_factors],
                      missing='drop').fit()

We can see that the coefficient of the Treatment (-3.1%) didn't change much as expected with the additional 
control variables.

print(multiple_reg.summary().tables[1])

## Exercises

1| In the literature of racial discrimination, there are more than 1000 observational studies for each experimental study. Suppose you read 100 observational studies that indicate that racial discrimination is real. Suppose that you also read 1 experimental study that claims no evidence of racial discrimination. Are you more inclined to accept the result of 100 observational studies or the result of the experimental study? Justify your answer.

2| Interpret the 4 values of the contingency table below. Specifically, state the meaning and compare the values.

The variable 'h': 1 = higher quality curriculum vitae; 0 = lower quality curriculum vitae. This variable was randomized.

Other variables were previously defined.

contingency_table = pd.crosstab(df['Treatment'], df['h'], 
                                values=df['call'], aggfunc='mean')
contingency_table

3| I created an interaction variable 'h_Treatment' that is the pairwise multiplication of the variable 'h' and 'treatment'. 

How can you use the coefficients of the regression below to get the values of the contingency table in exercise 2? Show the calculations.

df['h_Treatment'] = df['h']*df['Treatment']
interaction = sm.OLS(df['call'],
                      df[['Intercept', 'Treatment', 'h', 'h_Treatment'] ],
                      missing='drop').fit()
print(interaction.summary().tables[1])                      

4| I run the regression below without the interaction term 'h_Treatment'. Could I use the coefficients below to get the values of the contingency table in exercise 2?  If yes, show the exact calculations.

interaction = sm.OLS(df['call'],
                      df[['Intercept', 'Treatment', 'h'] ],
                      missing='drop').fit()
print(interaction.summary().tables[1])    

5| Write a code to get a contingency table below:


|firstname h |    0.0   |   1.0    |
|------------|----------|----------|
|Aisha       | 0.010000 | 0.037500 |
|Allison     | 0.121739 | 0.068376 |
|...         |    ...   |     ...  |

Inside the table are the callback rates broken by Curriculum Vitae quality.
What is the callback rate for Kristen and Lakisha? Why the rates are so different? Could we justify the rate difference, arguing that one is more educated and qualified than other? 

## Reference

Bertrand, Marianne, and Sendhil Mullainathan. (2004). Are Emily and Greg More Employable Than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination. American Economic Review, 94 (4): 991-1013. [Click to download the paper](https://github.com/causal-methods/Papers/raw/master/Are%20Emily%20and%20Greg%20More%20Employable%20than%20Lakisha%20and%20Jamal.pdf)
