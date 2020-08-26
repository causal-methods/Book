# 5) Could the Federal Reserve Prevent the Great Depression?

Vitor Kamada

E-mail: econometrics.methods@gmail.com

Last updated: 8-14-2020

Concerned the Great Depression, neoclassical economists tend to believe that the decline of economic activity led to bank failures. Keynes (1936) believed the opposite, that is, the bank insolvencies led to business failures.

Richardson & Troost (2009) noticed that during the Great Depression, the state of Mississippi was divided into two districts controlled by different branches of Federal Reserve (Fed): St. Louis and Atlanta. 

Differently from the St. Louis Fed that made more onerous to borrow money, the Atlanta Fed adopted a Keynesian policy of discount lending and emergency liquidity to illiquid banks.

Let's open the data from Ziebarth (2013). Each row is a firm from the Census of Manufactures (CoM) for 1929, 1931, 1933, and 1935.

# Load data from Ziebarth (2013)
import numpy as np
import pandas as pd
path = "https://github.com/causal-methods/Data/raw/master/" 
data = pd.read_stata(path + "MS_data_all_years_regs.dta")
data.head()

First, we must check how similar were the two districts St. Louis and Atlanta in 1929. All variables are reported in logarithms. The mean revenue of the firms in St. Louis district was 10.88; whereas the mean revenue of the firms in Atlanta district was 10.78. Both St. Louis and Atlanta had similar wage earners (4.54 vs 4.69) and hours per worker (4.07 vs 4) as well.

# Restrict the sample to the year: 1929
df1929 = data[data.censusyear.isin([1929])]

vars= ['log_total_output_value', 'log_wage_earners_total',
       'log_hours_per_wage_earner']

df1929.loc[:, vars].groupby(df1929["st_louis_fed"]).agg([np.size, np.mean])

Additionally, both St. Louis and Atlanta have a similar mean price (1.72 vs 1.55) and mean quantity (8.63 vs 8.83), if the 
sample is restricted to firms with 1 product.  Therefore, Atlanta district is a reasonable control group for St. Louis district.




# Restrict sample to firms with 1 product
df1929_1 = df1929[df1929.num_products.isin([1])]

per_unit = ['log_output_price_1', 'log_output_quantity_1']

df1929_1.loc[:, per_unit].groupby(df1929_1["st_louis_fed"]).agg([np.size, np.mean])

We want to see if the credit constrained policy of St. Louis Fed decreased the revenue of the firms. Or, in other words, if the Atlanta Fed saved firms from bankruptcy.

For this purpose, we have to explore the time dimension: the comparison of the firm revenue before 1929 and after 1931. 

Let's restrict the sample to the years 1929 and 1931. Then, let's drop the missing values.

# Restrict the sample to the years: 1929 and 1931
df = data[data.censusyear.isin([1929, 1931])]

vars = ['firmid', 'censusyear', 'log_total_output_value',
        'st_louis_fed', 'industrycode', 'year_1931']

# Drop missing values        
df = df.dropna(subset=vars)

Now, we can declare a panel data structure, that is, to set the unit of analysis and the time dimension. See that the variables "firmid" and "censusyear" became indices in the table. The order matters. The first variable must be the unit of analysis and the second variable must be time unit. See in the table that the firm (id = 12) is observed for two years: 1929 and 1931.

Note that panel data structure was declared after cleaning the data set. For example, if the missing values is dropped after the panel data declaration, the commands for regression will probably return errors.

df = df.set_index(['firmid', 'censusyear'])
df.head()



Let $Y_{it}$ the outcome variable of unit $i$ on time $t$. The dummy variable $d2_{it}$ is 1 for the second period, and 0 for the first period. Note that the explanatory variable $X_{it}$ varies over unit $i$ and time $t$, but the unobserved factor $\alpha_i$ doesn't vary over the time. Unobserved factor is an unavailable variable (data) that might be correlated with the variable of interest, generating bias in the results. 

$$Y_{it}=\beta_0+\delta_0d2_{it}+\beta_1X_{it}+\alpha_i+\epsilon_{it}$$

The advantage of exploring the time variation is that the unobserved factor $\alpha_i$ can be eliminated by a First-Difference (FD) method.

In the second period ($t=2$), the time dummy $d2=1$:

$$Y_{i2}=\beta_0+\delta_0+\beta_1X_{i2}+\alpha_i+\epsilon_{i2}$$

In the first period ($t=1$), the time dummy $d2=0$:

$$Y_{i1}=\beta_0+\beta_1X_{i1}+\alpha_i+\epsilon_{i1}$$

Then:

$$Y_{i2}-Y_{i1}$$

$$=\delta_0+\beta_1(X_{i2}-X_{i1})+\epsilon_{i2}-\epsilon_{i1}$$

$$\Delta Y_i=\delta_0+\beta_1\Delta X_i+\Delta \epsilon_i$$

Therefore, if the same units are observed over time (panel data), no need to worry about a factor that can be considered constant over the time analyzed. We can assume that the company culture and institutional practices don't vary much over a short period of time. These factors are likely to explain the difference in revenue among the firms but will not bias the result if the assumption above is correct.



Let's install the library that can run the panel data regressions.

!pip install linearmodels

Let's use the Difference-in-Differences (DID) method to estimate the impact of St. Louis Fed policy on firm revenue. In addition to explore the time difference, the treatment-control difference must be used to estimate the causal impact of the policy.

Let $Y$ be the outcome variable 'log_total_output_value', $d2$ the time dummy variable 'year_1931', $dT$ the treatment dummy variable 'st_louis_fed', and $d2 \cdot dT$ the interaction term between the previous two dummies:

$$Y = \beta_0+\delta_0d2+\beta_1dT+\delta_1 (d2\cdot dT)+ \epsilon$$

The DID estimator is given by $\delta_1$ and not by $\beta_1$ or $\delta_0$. First, we take the difference between "Treatment (St. Louis)" and "Control (Atlanta)", and then we take the difference between "After (1931)" and "Before (1921)". 

$$\hat{\delta}_1 = (\bar{y}_{2,T}-\bar{y}_{2,C})-(\bar{y}_{1,T}-\bar{y}_{1,C})$$

The order doesn't matter. If we take first the difference between "After (1931)" and "Before (1921)", and then the difference between "Treatment (St. Louis)" and "Control (Atlanta)", the result will be the same $\delta_1$. 

$$\hat{\delta}_1 = (\bar{y}_{2,T}-\bar{y}_{1,T})-(\bar{y}_{2,C}-\bar{y}_{1,C})$$

Let's show formally that the we must take the difference twice in the DID estimator $\delta_0$:  

If $d2=0$ and $dT=0$, then $Y_{0,0}=\beta_0$.

If $d2=1$ and $dT=0$, then $Y_{1,0}=\beta_0+\delta_0$.

$$Y_{1,1}=Y_{1,0}-Y_{0,0}=\delta_0$$

![alt text](https://github.com/causal-methods/Data/raw/master/figures/DiffDiff.PNG)

Let's manually calculate the $\hat{\delta}_1$ from the numbers in the graphic "Firm's Revenue during the Great Depression". Note that in the  Difference-in-Differences (DID) method, a counterfactual is constructed based on the control group (Atlanta). It is just a parallel shift of Atlanta line. The counterfactual is the hypothetical outcome for the treatment group (St. Louis), if St. Louis Fed had followed the same policy of Atlanta Fed.

$$\hat{\delta}_1 = (\bar{y}_{2,T}-\bar{y}_{2,C})-(\bar{y}_{1,T}-\bar{y}_{1,C})$$

$$=(10.32-10.42)-(10.87-10.78)$$

$$=(-0.1)-(-0.1)$$

$$=-0.2$$

The restrictive credit policy of St. Louis Fed decreased in about 20% the revenue of the firms. The result of a simple mean comparison in the end of 1931 is only -10%. Therefore, without using the counterfactual reasoning, the negative impact of the St. Louis Fed policy would be large underestimated.

# Mean Revenue for the Graphic
table = pd.crosstab(df['year_1931'], df['st_louis_fed'], 
        values=df['log_total_output_value'], aggfunc='mean')

# Build Graphic
import plotly.graph_objects as go
fig = go.Figure()

# x axis
year = [1929, 1931]

# Atlanta Line
fig.add_trace(go.Scatter(x=year, y=table[0],
                         name='Atlanta (Control)'))
# St. Louis Line
fig.add_trace(go.Scatter(x=year, y=table[1],
                         name='St. Louis (Treatment)'))
# Counterfactual
end_point = (table[1][0] - table[0][0]) + table[0][1]
counter = [table[1][0], end_point]
fig.add_trace(go.Scatter(x=year, y= counter,
                         name='Counterfactual',
                         line=dict(dash='dot') ))

# Difference-in-Differences (DID) estimation
fig.add_trace(go.Scatter(x=[1931, 1931],
                         y=[table[1][1], end_point],
                         name='$\delta_1=0.2$',
                         line=dict(dash='dashdot') ))

# Labels
fig.update_layout(title="Firm's Revenue during the Great Depression",
                  xaxis_type='category',
                  xaxis_title='Year',
                  yaxis_title='Log(Revenue)')

fig.show()

The result of Difference-in-Differences (DID) implemented via regression is:

$$\hat{Y} = 10.8-0.35d2+0.095dT-0.20(d2\cdot dT)$$

from linearmodels import PanelOLS

Y = df['log_total_output_value']
df['const'] = 1
df['louis_1931'] = df['st_louis_fed']*df['year_1931']

## Difference-in-Differences (DID) specification
dd = ['const', 'st_louis_fed', 'year_1931', 'louis_1931']

dif_in_dif = PanelOLS(Y, df[dd]).fit(cov_type='clustered',
                                     cluster_entity=True)
print(dif_in_dif)

The St. Louis Fed policy decreased the firm revenue in 18% ($1-e^{-0.1994}$). However, the p-value is 0.1074. The result is not statistically significant at 10%. 

from math import exp
1 - exp(dif_in_dif.params.louis_1931 )

Somebody might argue that the difference among firms is a confound factor. One or another big firm might bias the results.

This issue can be addressed by using Fixed Effects (FE) or Within Estimator. The technique is similar to the First-Difference (FD), but with different data transformation. The time-demeaned process is used to eliminate the unobserved factor $\alpha_i$. 

$$Y_{it}=\beta X_{it}+\alpha_i+\epsilon_{it}$$

Let's average the variables for each $i$ over time $t$:

$$\bar{Y}_{i}=\beta \bar{X}_{i}+\alpha_i+\bar{\epsilon}_{i}$$

Then, we take the difference and the unobserved factor $\alpha_i$ vanishes:

$$Y_{it}-\bar{Y}_{i}=\beta (X_{it}-\bar{X}_{i})+\epsilon_{it}-\bar{\epsilon}_{i}$$

We can write the equation above in a more compact way:

$$\ddot{Y}_{it}=\beta \ddot{X}_{it}+\ddot{\epsilon}_{it}$$

The computer implements the Fixed Effects (FE) automatically with the command "entity_effects=True".

As we declared previously that the firm is the unit of analysis in this panel data set, the computer implements the Firm Fixed Effects (FE) automatically with the command "entity_effects=True".

We added Firm Fixed Effects (FE) to the Difference-in-Differences (DID) specification and the result didn't change much. The intuition is that Difference-in-Differences (DID) technique had already mitigated the endogeneity problems.

The St. Louis Fed policy decreased the firm revenue in 17% ($1-e^{-0.1862}$). The result is statistically significant at 10%.




firmFE = PanelOLS(Y, df[dd], entity_effects=True)
print(firmFE.fit(cov_type='clustered', cluster_entity=True))

The Fixed Effects (FE) can be manually implemented by adding dummy variables. Let's add Industry Fixed Effects to the Difference-in-Differences (DID) specification to discard the possibility that the result might be driven by Industry specific shocks.

The St. Louis Fed policy decreased the firm revenue in 14.2% ($1-e^{-0.1533}$). The result is statistically significant at 10%.

Why not add Firm and Industry Fixed Effects in the same time? It is possible and recommendable, but the computer will not return any result given the problem of multicollinearity. We have only two observations (2 years) per firm. If we add one dummy variable for each firm, it is like to run a regression with more variables than observations.

In his paper Ziebarth (2013) presents results using Firm and Industry Fixed Effects, how is it possible? Ziebarth (2013) used Stata software. Stata automatically drops some variables in the case of multicollinearity problem and outputs a result. Although this practice is well-diffused in Top Journals of Economics, it is not the "true" Fixed Effects.

industryFE = PanelOLS(Y, df[dd + ['industrycode']])
print(industryFE.fit(cov_type='clustered', cluster_entity=True))

Just for exercise purpose, suppose that the unobserved factor $\alpha_i$ is ignored. This assumption is called Random Effects (RE). In this case, $\alpha_i$ will be inside the error term $v_{it}$ and potentially biased the results.

$$Y_{it}=\beta X_{it}+v_{it}$$

$$v_{it}= \alpha_i+\epsilon_{it}$$

In an experiment, the treatment variable is uncorrelated with the unobserved factor $\alpha_i$. In this case, Random Effects (RE) model has the advantage of producing lower standard errors than the Fixed Effects models.

Note that if we run a simple Random Effects (RE) regression, we might conclude wrongly that St. Louis Fed policy increased the firm revenue in 7%.

from linearmodels import RandomEffects
re = RandomEffects(Y, df[['const', 'st_louis_fed']])
print(re.fit(cov_type='clustered', cluster_entity=True))

## Exercises

1| Suppose a non-experimental setting, where the control group differs from the treatment group. Justify if it is reasonable or not to use Difference-in-Differences (DID) to estimate a causal effect? Should you modify or add something in the DID framework?


2| Suppose a study claims based on Difference-in-Differences (DID) method that Fed avoided massive business failures via the bank bailout of 2008. Suppose another study based on Regression Discontinuity (RD) claims the opposite or denies the impact of Fed on business failures. What do you think is more credible empirical strategy DID or RD to estimate the causal impact of Fed policy? Justify you answer.


3| In a panel data, where the unit of analysis can be firm or county, what is more credible the result at firm or at county level? Justify.

4| Use the data from Ziebarth (2013) to estimate the impact of St. Louis Fed policy on firm's revenue. Specifically, run Difference-in-Differences (DID) with Random Effects (RE). Interpret the result. What can be inferred about the unobserved factor $\alpha_i$? 

5| Use the data from Ziebarth (2013) to estimate the impact of St. Louis Fed policy on firm's revenue. Specifically, run Difference-in-Differences (DID) with Firm Fixed Effects (FE) without using the command "entity_effects=True". Hint: You must use dummy variables for each firm.

## Reference

Keynes, John Maynard. (1936). The General Theory of Employment, Interest and Money. Harcourt, Brace and Company, and printed in the U.S.A. by the Polygraphic Company of America, New York. [Click to access the book](https://www.marxists.org/reference/subject/economics/keynes/general-theory/) 

Richardson, Gary, and William Troost. (2009). Monetary Intervention Mitigated Banking Panics during the Great Depression: Quasi-Experimental Evidence from a Federal Reserve District Border, 1929-1933. Journal of Political Economy 117 (6): 1031-73. [Click to download the paper](https://github.com/causal-methods/Papers/raw/master/richardson_troost_2009_jpe.pdf) 

Ziebarth, Nicolas L. (2013). Identifying the Effects of Bank Failures from a Natural Experiment in Mississippi during the Great Depression. American Economic Journal: Macroeconomics, 5 (1): 81-101. [Click to download the paper](https://github.com/causal-methods/Papers/raw/master/Identifying%20the%20Effects%20of%20Bank%20Failures.pdf) 