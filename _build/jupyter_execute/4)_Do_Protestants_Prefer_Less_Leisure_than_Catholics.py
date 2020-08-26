# 4) Do Protestants Prefer Less Leisure than Catholics?  

Vitor Kamada

E-mail: econometrics.methods@gmail.com

Last updated: 8-11-2020

Max Weber (1930) argued that the Protestant ethic, especially the variants of Calvinism is more aligned with capitalism than the Catholicism. Weber (1930) observed that Protestant regions in Northern Europe were more developed than the Catholic regions in the Southern Europe. He hypothesized that Protestants work hard and more, save more, rely more on themselves, and expect less from the Government. All characteristics that would lead to greater economic prosperity. 
  
Maybe, it is not the religion the cause of great economic performance. Education is a confound factor. Historically, Protestants have higher level of literacy, because they were incentivized to read the Bible. 

The causal effect can be reverse as well. Perhaps, an industrial person is more likely to become Protestant. Religion is a choice variable. People self-select the ideology that confirms their own view of the world.

Let's open the data from Basten & Betz (2013). Each row represents a municipality in Western Switzerland, the cantons of Vaud and Fribourg.

# Load data from Basten & Betz (2013)
import numpy as np
import pandas as pd
path = "https://github.com/causal-methods/Data/raw/master/" 
df = pd.read_stata(path + "finaldata.dta")
df.head(4)

Switzerland is a very diverse country in terms of geography and institutions. It is not fair to compare a rural Catholic that lives in the Alpes with an urban high educated Protestant in Zurich.

Historically, the cities had different incentives to adopt Protestantism or remain Catholic. Cities with a stronger merchant guild were more likely to adopt the Protestantism; whereas cities governed by aristocrats were more likely to remain Catholic.

There are too much confound factors to use the whole country. The analysis will be restricted to the cantons of Vaud (historically Protestant) and Fribourg (historically Catholic). See the map below from Basten & Betz (2013). 

This region of 4,883 $km^2$, that represents 4.5 percent of Switzerland, is institutionally and geographically homogeneous. In 1536, the canton of Vaud didn't self-select to become Protestant, but it was forced because of a war. Therefore, this is a quasi-experiment setting, where treatment and control region is similar to each other, because of a historical event.


![alt text](https://github.com/causal-methods/Data/raw/master/figures/CatholicProtestant.PNG)

**Source:** Basten & Betz (2013)

In the graphic below, we can see that higher the proportion of protestants in a municipality, lower the preference for leisure. The blue dots are historically Catholic municipalities (Fribourg), while the
red dots are historically Protestant municipalities (Vaud). It looks like distinct subgroups. How can we figure out if there is evidence of causal effect or it is a mere correlation?

# Create variable "Region" for the graphic
def category(var):
    if var == 1:
        return "Protestant (Vaud)"
    else:   
        return "Catholic (Fribourg)"
df['Region'] = df["vaud"].apply(category)

# Rename variables with auto-explanatory names
df = df.rename(columns={"prot1980s": "Share_of_Protestants",
                        "pfl": "Preference_for_Leisure"})

# Scatter plot
import plotly.express as px
leisure = px.scatter(df,
                     x="Share_of_Protestants",
                     y="Preference_for_Leisure",
                     color="Region")
leisure.show()

Let's refine the analysis. Remember that Regression Discontinuity is the closer technique to an experiment.

In the graphic below, there is a discontinuity in the preference for leisure at border distance = 0. The border distance above 0 encompasses historically Protestant municipalities (Vaud); whereas, the border distance below 0 encompasses historically Catholic municipalities (Fribourg).

The running variable "Border Distance" determines the region, but not the share of protestants. However, the share of protestants increases as function of the distance.

df = df.rename(columns={"borderdis": "Border_Distance_in_Km"})

leisure = px.scatter(df,
                     x="Border_Distance_in_Km",
                     y="Preference_for_Leisure",
                     color="Share_of_Protestants",
                     title="Discontinuity at Distance = 0")
leisure.show()

As the border is arbitrary, that is, determined by a historic event, the municipalities closer to the border is likely to be more similar to each other than the municipalities far away of the border. Therefore, let's restrict the analysis to municipalities inside a range of 5 Km.   

df5 = df[df['Border_Distance_in_Km'] >= -5]
df5 = df5[df5['Border_Distance_in_Km'] <= 5]

The simple mean comparison shows that the preference for leisure is lower in the Protestant municipalities (39.5%) compared with the Catholic municipalities (48.2%). The difference is -8.7%.

Note that the Protestant region has higher mean income measure in Swiss Franc (47.2K vs 43.7K) and higher inequality captured by Gini index (0.36 vs 0.30).


df5 = df5.rename(columns={"reineink_pc_mean": "Mean_Income_(CHF)",
                          "Ecoplan_gini"    : "Gini_1996"})
                          
outcome = ['Preference_for_Leisure', 'Mean_Income_(CHF)', 'Gini_1996']
df5.loc[:, outcome].groupby(df5["Region"]).agg([np.size, np.mean])

The comparison above is not "bad", considering that only the municipalities inside a 5 Km range is used (49 Catholic and 84 Protestant municipalities). Furthermore, the two regions are similar in terms of share of no religious affiliation in 1980 (1.7% vs 2.9%) and altitude above the sea level (642 vs 639 meters).

However, a more credible approach is to use a regression discontinuity framework with the running variable ($r_i$). 

control = ['noreligion1980s', 'altitude']
df5.loc[:, control].groupby(df5['Region']).agg([np.mean, np.std])

In the graphic "Fuzzy Regression Discontinuity", we can clearly see that the running variable "Border Distance" is very correlated with the treatment variable "Share of Protestants". The variable "Border Distance" does not determine the treatment status but increases the probability of being Protestants. Therefore, this is a case of a Fuzzy Regression Discontinuity and not a Sharp Regression Discontinuity.

Let $D_i$ be the treatment status of unit $i$. $P(D_i=1|r_i)$ is a jump in the probability of treatment at cutoff $r_0$:   

$$P(D_i=1|r_i)$$

$$= f_1(r_i) \ if \ r_i\geq r_0$$

$$= f_0(r_i) \ if \ r_i< r_0$$

where $f_1(r_i)$ and $f_0(r_i)$ are functions that can assume any value. In the Sharp Regression Discontinuity, $f_1(r_i)$ was 1 and $f_0(r_i)$ was 0.


fuzzy = px.scatter(df5,
                     x="Border_Distance_in_Km",
                     y="Share_of_Protestants",
                     color="Region",
                     title="Fuzzy Regression Discontinuity")
fuzzy.show()

In the graphic below, the variable share of protestants is simulated to illustrate what would be a case of Sharp Regression Discontinuity. 

def dummy(var):
    if var >= 0:
        return 1
    else:   
        return 0

df5["Simulated_Share_Protestant"] = df5["Border_Distance_in_Km"].apply(dummy)

sharp = px.scatter(df5,
                     x="Border_Distance_in_Km",
                     y="Simulated_Share_Protestant",
                     color="Region",
                     title="Sharp Regression Discontinuity")
sharp.show()



Let's assume $Y$ =  Preference_for_Leisure, $D_r$ = Share_of_Protestants, and $r$ = Border_Distance. Now, we have a problem to estimate the equation below: 

$$Y = \beta_0+\rho D_r+ \beta_1r+\epsilon$$

The variable of interest $D_r$ is not anymore "purified" by $r$, that is, it is not anymore completely determined by the running variable $r$. Therefore, $D_r$ is likely to be correlated with the error term $\epsilon$:

$$Cov(D_r, \epsilon)\neq0$$

We can fix this problem using an instrumental variable $Z$ that is uncorrelated with the error term $\epsilon$, and correlated with $D_r$ after controlling for other factors:

$$Cov(Z, \epsilon) = 0$$

$$Cov(Z, D_r) \neq 0$$

The natural candidate for $Z$ is the variable "vaud": 1 if it is a municipality in Vaud; and 0 if it is a municipality in Fribourg. There is no reason to believe that this variable is correlated with the error term $\epsilon$, as the border that divides the region was determined in 1536, when the Republic of Berne conquered Vaud. The second assumption is also valid, as higher share of Protestants live in the Vaud region than Fribourg.

The instrumental variable method consists first in "purifying" $D_r$ using $Z$:

$$D_r=\gamma_0+\gamma_1Z+\gamma_2r+\upsilon$$

Then, we get the fitted values of $\hat{D}_r$ by running an ordinary least square (OLS) and plug it in the equation:

$$Y = \beta_0+\rho \hat{D}_r+ \beta_1r+\epsilon$$

The logic is that the "purified" $\hat{D}_r$ is uncorrelated with the error term $\epsilon$. Now, we can run an ordinary least square (OLS) to get the isolated effect of $\hat{D}_r$ on $Y$, that is, $\rho$ will be the unbiased causal effect.

# Install libray to run Instrumental Variable estimation
!pip install linearmodels

The computer automatically run the two stages of Instrumental Variable (IV) procedure. We indicated that the endogenous variable $D_r$ is "Share of Protestants", and the instrument variable $Z$ is "vaud". We also add the control variable "t_dist" that is the interaction between the variables "vaud" and "Border Distance".

The result is that "Share of Protestants" decreases the preference for leisure in 13.4%. 

from linearmodels.iv import IV2SLS
iv = 'Preference_for_Leisure ~ 1 + Border_Distance_in_Km + t_dist + [Share_of_Protestants ~ vaud]'
iv_result = IV2SLS.from_formula(iv, df5).fit(cov_type='robust')

print(iv_result)

We can also check the first stage to see if the instrumental variable "vaud" is correlated with "Share of Protestants" after controlling for other factors like "Border Distance" and "t_dist". Vaud increases 67% the share of Protestants. The t-value of "vaud" is 20, that is, statistically significant without any margin of doubt. 

Therefore, we are confident that the second stage result is more credible than the simple mean comparison. The Instrumental Variable impact of 13.4% is more credible than the simple mean comparison of 8.7%.

print(iv_result.first_stage)

The simple mean comparison result of 8.7% is closer to the result of 9% from the naive Sharp Regression Discontinuity (SRD) below. The Vaud region has a 9% less preference for leisure than the Fribourg. We cannot conclude that Protestants have a 9% less preference for leisure than Catholics. The Vaud region is not 100% Protestant. Neither the Fribourg region is 100% Catholic. 

The Fuzz Regression Discontinuity (FRD), that uses the Instrumental Variable (IV) estimation, is a correction for the naive comparison. The FRD isolates the impact of Protestants on preference for leisure. Therefore, the most credible estimation is that Protestants have 13.4% less preference for leisure than Catholics.

naive_srd = 'Preference_for_Leisure ~ 1 + vaud + Border_Distance_in_Km + t_dist'
srd = IV2SLS.from_formula(naive_srd, df5).fit(cov_type='robust')
print(srd)

## Exercises

1| What would be the confound factors in estimating the causal impact of Protestants against Catholics on economic prosperity of Switzerland (whole country)? Explain in detail each confound factor.

2| Somebody could argue that Western Switzerland was a diverse region before 1536, when the Republic of Berne conquered Vaud. In this case, Fribourg (Catholic) would not be a reasonable control group for Vaud (Protestant). What variables could be used to test the homogeneity/diversity of the region before 1536? Indicate if the data exists, or if it is feasible to collect the data.

3| I replicated the main results of the paper  Basten and Betz (2013) and I noticed that they don't control for education. Becker & Woessmann(2009) argue that Protestantism effect on the economic prosperity of Prussia in nineteenth century is due to higher literacy. Basten and Betz (2013) present the table below in the [Online Appendix](https://github.com/causal-methods/Papers/raw/master/Beyond-Work-Ethic/2011-0231_app.pdf). Do the numbers in the table strengthen or weaken the results of Basten and Betz (2013)? Justify your reasoning.    

![alt text](https://github.com/causal-methods/Data/raw/master/figures/PIsaScore.PNG)

**Source:** Basten & Betz (2013)

4| Preference for leisure is a self-reported variable in a survey. Maybe, Protestants have more incentives to declare lower preference for leisure than Catholics. In the real word, Protestants might enjoy leisure as much as Catholics. Declared preference might not match with real behavior. The relevant reseach question is if the religion (Protestant) causes people to be hard-working in actuality. Be creative and propose a way (methods and data) to fix the problem described above.

5| Use the data from Basten and Betz (2013) to investigate if Protestants cause higher mean income in Western Switzerland. Adopt the specifications that you think it is the most credible to recover the unbiased causal effect. Explain and justify each step of your reasoning. Interpret the main results.


## Reference

Basten, Christoph, and Frank Betz (2013). Beyond Work Ethic: Religion, Individual, and Political Preferences. American Economic Journal: Economic Policy, 5 (3): 67-91. [Click to download the paper](https://github.com/causal-methods/Papers/raw/master/Beyond-Work-Ethic/Beyond%20Work%20Ethic.pdf) and [Online Appendix](https://github.com/causal-methods/Papers/raw/master/Beyond-Work-Ethic/2011-0231_app.pdf)

Becker, Sascha O., and Ludger Woessmann. (2009). Was Weber Wrong? A Human Capital Theory of Protestant Economic History. Quarterly Journal of Economics 124 (2): 531–96.

Weber, Max. (1930). The Protestant Ethic and the Spirit of Capitalism. New York: Scribner, (Orig.pub. 1905).