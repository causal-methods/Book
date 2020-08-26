# 3) Are Females More Likely to Complete High School Under Islamic or Secular Regime?

Vitor Kamada

E-mail: econometrics.methods@gmail.com

Last updated: 8-8-2020

Let's open the data from Meyersson (2014). Each row represents a municipality in Turkey. 

# Load data from Meyersson (2014)
import numpy as np
import pandas as pd
path = "https://github.com/causal-methods/Data/raw/master/" 
df = pd.read_stata(path + "regdata0.dta")
df.head()

The variable 'hischshr1520f' is the proportion of female aged 15-20 that completed high  according to the 2000 census. Unfortunately, the age is aggregated. It is unlikely that 15 and 16 year old teenagers can finish high school in Turkey. It would be better to have the data broken by age. As the 15 and 16 year old cannot be removed from the analysis, the proportion of female aged 15-20 that completed high school is very low: 16.3%. 

The variable 'i94' is 1 if an Islamic mayor won the municipality election in 1994, and 0 if a secular mayor won. The Islamic party governed 12% of the municipalities in Turkey.

# Drop missing values
df = df.dropna(subset=['hischshr1520f', 'i94'])

# Summary Statistics
df.loc[:, ('hischshr1520f', 'i94')].describe()[0:3]

The average high school attainment for the females aged 15-20 is 14% in the municipalities governed by an Islamic major versus 16.6% in the municipalities governed by a secular major.

This is a naive comparison, because the data is not from an experiment. The mayor type was not randomized and cannot be randomized in practice. For example, poverty might lead to a higher level of religiosity and a lower educational achievement. It might be poverty that causes lower rate of high school attainment rather than religion. 

df.loc[:, ('hischshr1520f')].groupby(df['i94']).agg([np.size, np.mean])

The graphic "Naive Comparison" shows that control and treatment group is determined based on the variable 'iwm94': Islamic win margin. This variable was centralized to 0. Therefore, if win margin is above 0, the Islamic mayor won on the election. By the other hand, if win margin is below 0, the Islamic mayor lost the election.

In terms of average high school attainment, the difference between treatment group (14%) and control group (16.6%) is -2.6%. The problem of comparing municipal outcomes using observational data is that the treatment group is not similar to the control group. Therefore, confound factors might bias the results. 



import matplotlib.pyplot as plt

# Scatter plot with vertical line
plt.scatter(df['iwm94'], df['hischshr1520f'], alpha=0.2)
plt.vlines(0, 0, 0.8, colors='red', linestyles='dashed')

# Labels
plt.title('Naive Comparison')
plt.xlabel('Islamic win margin')
plt.ylabel('Female aged 15-20 with high school')

# Control vs Treatment
plt.text(-1, 0.7, r'$\bar{y}_{control}=16.6\%$', fontsize=16,
         bbox={'facecolor':'yellow', 'alpha':0.2})
plt.text(0.2, 0.7, r'$\bar{y}_{treatment}=14\%$', fontsize=16,
         bbox={'facecolor':'yellow', 'alpha':0.2})
plt.show()

This 2.6% difference between high school attainment governed by an Islamic major and a secular major is statistically significant at 1% level of significance. The magnitude is also relevant given that the mean value of high school completion is 16.3%. However, note that it is a naive comparison and likely to be biased. 

# Naive Comparison
df['Intercept'] = 1
import statsmodels.api as sm
naive = sm.OLS(df['hischshr1520f'], df[['Intercept', 'i94']],
                    missing='drop').fit()
print(naive.summary().tables[1])

One way to figure out if the naive comparison is likely to be biased is to check if the municipalities ruled by the Islamic major is different from the municipalities ruled by a secular major. 

The municipalities, where the Islamic major won, have higher Islamic vote share in 1994 (41% vs 10%), bigger number of parties receiving votes (5.9 vs 5.5), bigger log population (8.3 vs 7.7), higher population share below 19 year old (44% vs 40%), bigger household size (6.4 vs 5.75), higher proportion of district center (39% vs 33%), and higher proportion of province center (6.6% vs 1.6%).

df = df.rename(columns={"shhs"   : "household",
                        "merkezi": "district",
                        "merkezp": "province"})

control = ['vshr_islam1994', 'partycount', 'lpop1994',
           'ageshr19', 'household', 'district', 'province']
df.loc[:, control].groupby(df['i94']).agg([np.mean])

One way to make control and treatment group similar to each other is to use multiple regression. The interpretation of the coefficient of the treatment variable 'i94' is *ceteris paribus*, that is, the impact of Islamic major on high school attainment considering everything else constant. The trick here is the "everything else constant" that means only the factors that is controlled in the regression. This is an imperfect solution, because in practical terms, it is not possible to control for all factors that affect the outcome variable. However, compared to the simple regression, the multiple regression is likely to suffer less from the omitted variable bias. 

The multiple regression below challenges the result of the naive comparison. Islamic regime has a positive impact of 1.4% higher high school completion compared with a secular regime. The result is statistically significant at 5%.

multiple = sm.OLS(df['hischshr1520f'],
                      df[['Intercept', 'i94'] + control],
                      missing='drop').fit()
print(multiple.summary().tables[1])                      

The result of the multiple regression looks counter intuitive. How the sign of the treatment variable can change? 

Let's look at data from other perspective. The graph "Naive Comparison" is the scatterplot of all municipalities individually. It is hard to see any pattern or trends. 

Let's plot the same graphic, but with municipalities aggregated in 29 bins based on similarity of the outcome variable high school completion. These bins are the blue balls in the graphic below. The size of the ball is proportional to the number of municipalities used to calculate the mean value of high school completion. 

If you look carefully near the cut-off (vertical red line), where the variable Islamic win margin = 0, you will see a discontinuity or a jump in the level of high school completion. 

# Library for Regression Discontinuity
!pip install rdd

from rdd import rdd

# Aggregate the data in 29 bins
threshold = 0
data_rdd = rdd.truncated_data(df, 'iwm94', 0.99, cut=threshold)
data_binned = rdd.bin_data(data_rdd, 'hischshr1520f', 'iwm94', 29)

# Labels
plt.title('Comparison using aggregate data (Bins)')
plt.xlabel('Islamic win margin')
plt.ylabel('Female aged 15-20 with high school')

# Scatterplot 
plt.scatter(data_binned['iwm94'], data_binned['hischshr1520f'],
    s = data_binned['n_obs'], facecolors='none', edgecolors='blue')

# Red Vertical Line
plt.axvline(x=0, color='red')

plt.show()

Maybe you are not convinced that there is a discontinuity or a jump in the cut-off point. Let's plot the same graphic with 10 bins and restrict the bandwidth (range) of the variable Islamic win margin. Rather than choosing an arbitrary bandwidth (h), let's use a method developed by Imbens & Kalyanaraman (2012) to get the optimal bandwidth that minimizes the mean squared error.

The optimal bandwidth ($\hat{h}$) is 0.23, that is, let's get a window of 0.23 below and above the cut-off to create the 10 bins.

#  Optimal Bandwidth based on Imbens & Kalyanaraman (2012)
#  This bandwidth minimizes the mean squared error.
bandwidth_opt = rdd.optimal_bandwidth(df['hischshr1520f'],
                              df['iwm94'], cut=threshold)
bandwidth_opt

Below are the 10 bins. There are 5 bins in the control group, where the Islamic win margin < 0, and 5 bins in the treatment group, where the Islamic win margin > 0. See that high school completion jumps from 13.8% to 15.5% between index 4 and 5, respectively bins 5 and 6. The values 13.8% and 15.5% were computed based on respectively 141 and 106 municipalities ('n_obs').

#  Aggregate the data in 10 bins using Optimal Bandwidth
data_rdd = rdd.truncated_data(df, 'iwm94', bandwidth_opt, cut=threshold)
data_binned = rdd.bin_data(data_rdd, 'hischshr1520f', 'iwm94', 10)
data_binned

In the graphic "Comparison using Optimum Bandwidth (h = 0.27)", a blue line was fitted to the control group (5 bins), and an orange line was fitted to the treatment group (5 bins). Now, the discontinuity or jump is clear. We call this method Regression Discontinuity (RD).  The red vertical line ($\hat{\tau}_{rd}=3.5$%) is the increase of high school completion. Note that this method mimics an experiment. The municipalities where the Islamic party barely won and barely lost are likely to be similar to each other. The intuition is that "barely won" and "barely lost" is a random process like flip a coin. The reverse result in the election could occur at random. By the other hand, it is hard to imagine that Islamic mayors could lose in the 
municipalities where they won by a stronger margin of 30%.

# Scatterplot
plt.scatter(data_binned['iwm94'], data_binned['hischshr1520f'],
    s = data_binned['n_obs'], facecolors='none', edgecolors='blue')

# Labels
plt.title('Comparison using Optimum Bandwidth (h = 0.27)')
plt.xlabel('Islamic win margin')
plt.ylabel('Female aged 15-20 with high school')

# Regression
x = data_binned['iwm94']
y = data_binned['hischshr1520f']

c_slope , c_intercept = np.polyfit(x[0:5], y[0:5], 1)
plt.plot(x[0:6], c_slope*x[0:6] + c_intercept)

t_slope , t_intercept  = np.polyfit(x[5:10], y[5:10], 1)
plt.plot(x[4:10], t_slope*x[4:10] + t_intercept)

# Vertical Line
plt.vlines(0, 0, 0.2, colors='green', alpha =0.5)
plt.vlines(0, c_intercept, t_intercept, colors='red', linestyles='-')

# Plot Black Arrow
plt.axes().arrow(0, (t_intercept + c_intercept)/2, 
         dx = 0.15, dy =-0.06, head_width=0.02,
         head_length=0.01, fc='k', ec='k')

# RD effect
plt.text(0.1, 0.06, r'$\hat{\tau}_{rd}=3.5\%$', fontsize=16,
         bbox={'facecolor':'yellow', 'alpha':0.2})

plt.show()

# RD effect given by the vertical red line
t_intercept - c_intercept

Let's restrict the sample to the municipalities where the Islamic mayor won or lost by a margin of 5%. We can see that the control group and the treatment group are more similar to each other than the comparison using the full sample in the beginning of this chapter. 

However, this similarity is not closer to a "perfect experiment". Part of the reason is the small sample size of the control and treatment group. Therefore, when we run the Regression Discontinuity, it is advisable to add the control variables.

# bandwidth (h) = 5%
df5 = df[df['iwm94'] >= -0.05]
df5 = df5[df5['iwm94'] <= 0.05]

df5.loc[:, control].groupby(df5['i94']).agg([np.mean])

Let's formalize the theory of Regression Discontinuity.

Let the $D_r$ be a dummy variable: 1 if the unit of analysis receives the treatment, and 0 otherwise. The subscript $r$ indicates that the treatment ($D_r$) is a function of the running variable $r$.

$$D_r =  1 \ or \ 0$$

In the Sharp Regression Discontinuity, the treatment ($D_r$) is determined by the running variable ($r$).

$$D_r =  1, \ if \ r \geq r_0$$

$$D_r =  0, \ if \ r < r_0$$

where, $r_0$ is an arbitrary cutoff or threshold.

The most basic specification of the Regression Discontinuity is:

$$Y = \beta_0+\tau D_r+ \beta_1r+\epsilon$$

where $Y$ is the outcome variable, $\beta_0$ the intercept, $\tau$ the impact of the treatment variable ($D_r$), $\beta_1$ the coefficient of the running variable ($r$), and $\epsilon$ the error term.

Note that in an experiment, the treatment is randomized, but in Regression Discontinuity, the treatment is completely determined by the running variable. The opposite of a random process is a deterministic process. It is counter-intuitive, but the deterministic assignment has the same effect of randomization, when the rule (cutoff) that determines the treatment assignment is arbitrary. 

In general, the credibility of observational studies is very weak, because of the fundamental problem of the omitted variable bias (OVB). Many unobserved factors inside the error term might be correlated with the treatment variable. Therefore, the big mistake in a regression framework is to leave the running variable inside the error term.

Among all estimators, Regression Discontinuity is probably the closer method to the golden standard, randomized experiment. The main drawback is that Regression Discontinuity only captures the local average treatment effect (LATE). It is unreasonable to generalize the results to the entities outside the bandwidth. 

The impact of Islamic mayor is 4% higher female school completion, using a Regression Discontinuity with bandwidth of 5%. This result is statistically significant at level of significance 5%.

#  Real RD specification
#  Meyersson (2014) doesn't use the interaction term, because 
# the results are unstable. In general the coefficient,
# of the interaction term is not statistically significant.
# df['i94_iwm94'] = df['i94']*df['iwm94']
# RD = ['Intercept', 'i94', 'iwm94', 'i94_iwm94']

RD = ['Intercept', 'i94', 'iwm94']

# bandwidth of 5%
df5 = df[df['iwm94'] >= -0.05]
df5 = df5[df5['iwm94'] <= 0.05]
rd5 = sm.OLS(df5['hischshr1520f'],
                      df5[RD + control],
                      missing='drop').fit()
print(rd5.summary()) 

The impact of Islamic mayor is 2.1% higher female school completion, using a Regression Discontinuity with optimal bandwidth 0.27 calculated based on Imbens & Kalyanaraman (2012). This result is statistically significant at level of significance 1%.

Therefore, the Regression Discontinuity estimators indicates that the naive comparison is biased in the wrong direction.

# bandwidth_opt is 0.2715
df27 = df[df['iwm94'] >= -bandwidth_opt]
df27 = df27[df27['iwm94'] <= bandwidth_opt]
rd27 = sm.OLS(df27['hischshr1520f'],
                      df27[RD + control],
                      missing='drop').fit()
print(rd27.summary()) 

## Exercises

1| Use the data from Meyersson (2014) to run a Regression Discontinuity: a) with full sample, and b) another with bandwidth of 0.1 (10% for both sides). Use the same specification of the two examples of this chapter. Interpret the coefficient of the treatment variable. What is more credible the result of "a" or "b"? Justify.

2| Below is the histogram of the variable Islamic win margin. Do you see any discontinuity or abnormal pattern where the cutoff = 0? What is the rationality of investigating if something weird is going on around the cutoff of the running variable?

import plotly.express as px
fig = px.histogram(df, x="iwm94")
fig.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0 = 0, y1 = 1,
      xref= 'x', x0 = 0, x1 = 0)])
fig.show()

3| I modified the variable "Islamic win margin" for educational purpose. Suppose this is the real running variable from Meyersson (2014). See the histogram below. In this hypothetical situation, what can you infer about the elections in Turkey? Is there a problem in running Regression Discontinuity in this situation? If yes, what can you do to solve the problem?

def corrupt(variable):
    if variable <= 0 and variable >= -.025:
        return 0.025
    else:   
        return variable

df['running'] = df["iwm94"].apply(corrupt)

fig = px.histogram(df, x="running")
fig.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0 = 0, y1 = 1,
      xref= 'x', x0 = 0, x1 = 0)])
fig.show()

4| Explain the graphic below for somebody who is an expert in Machine Learning, but is not trained in Causal Inference? Could the variable "Islamic vote share" be used as running variable? Speculate.

def category(var):
    if var <= 0.05 and var >= -.05:
        return "5%"
    else:   
        return "rest"

df['margin'] = df["iwm94"].apply(category)

fig = px.scatter(df, x="vshr_islam1994", y="iwm94", color ="margin",
                 labels={"iwm94": "Islamic win margin",
                         "vshr_islam1994": "Islamic vote share"})
fig.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0 = 1/2, y1 = 1/2,
      xref= 'x', x0 = 0, x1 = 1)])
fig.show()

5| Are males more likely to complete high school under Islamic or secular regime? Justify your answer based on data and rigorous analysis. The variable "hischshr1520m" is the proportion of males aged 15-20 with high school education.

## Reference

Imbens, G., & Kalyanaraman, K. (2012). Optimal Bandwidth Choice for the Regression Discontinuity Estimator. The Review of Economic Studies, 79(3), 933-959.

Meyersson, Erik. (2014). Islamic Rule and the Empowerment of the Poor and Pious. Econometrica, 82(1), 229-269. [Click to download the paper](https://github.com/causal-methods/Papers/raw/master/Islamic%20Rule%20and%20the%20Empowerment%20of%20the%20Poor%20and%20Pious.pdf) 