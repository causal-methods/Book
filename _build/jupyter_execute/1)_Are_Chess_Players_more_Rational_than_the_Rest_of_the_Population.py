# 1) Are Chess Players More Rational than the Rest of the Population?

[Vitor Kamada](https://www.linkedin.com/in/vitor-kamada-1b73a078)

E-mail: econometrics.methods@gmail.com

Last updated: 8-26-2020

Look at the figure of the Centipedes game below from Palacios-Huerta & Volij (2009).
Each player has only two strategies: "Stop" or "Continue". If player 1 stops in the first round, he gets \$4 and player 2 gets \$1. If player 1 continues,  player 2 can stop or continue. If player 2 stops, he will receive \$8, and player 1 will receive \$2. If player 2 continues, player 1 again can decide between "Stop" or "Continue". From the social point of view, it is better that both players play "Continue" in the six rounds, then player 1 can receive \$256 and player 2 can receive \$64. However, if player 2 is rational, he will never play "Continue" in the round 6, because he can receive \$128, if he stops. Knowing that by backward induction, it is irrational to player 1 to continue in round 1. 

![alt text](https://github.com/causal-methods/Papers/raw/master/Centipedes/centipede.PNG)

**Source**: Palacios-Huerta & Volij (2009)

Several experimental studies show that almost nobody stops at the first opportunity. According with the table below from Palacios-Huerta & Volij (2009), only 7.5% of students stopped in the first round. The most part of students (35%) stopped in the round 3. Similar results with big sample size can be found in McKelvey & Palfrey (1992).

![alt text](https://github.com/causal-methods/Papers/raw/master/Centipedes/UPV.PNG)

**Source**: Palacios-Huerta & Volij (2009)

Let's open the data set from Palacios-Huerta & Volij (2009), containing information on how chess players played the Centipede game.

import pandas as pd
path = "https://github.com/causal-methods/Data/raw/master/" 
sheet_name = "Data Chess Tournaments"
chess = pd.read_excel(path + "Chess.xls", sheet_name)
chess

Let's count the number of chess players in each category. In the sample, we can see 26 Grandmasters (GM), 29 International Masters (IM), and 15 Federation Masters (FM). 

chess['Title1'].value_counts() 

Let's combine all other chess players under the label "Other". 

dictionary = {    'Title1':
               {    0: "Other",
                'WGM': "Other", 
                'WIM': "Other", 
                'WFM': "Other",
                 'CM': "Other" }}

chess.replace(dictionary, inplace=True)

We have 141 chess players in the category "Other".

chess['Title1'].value_counts() 

Let's focus on the title and rating of the chess players marked by 1. Let's ignore the title and rating of their opponents, that is, chess players marked by 2. 

title_rating1 = chess.filter(["Title1", "ELORating1"])
title_rating1 

We can see that the average rating of Grandmasters is 2513, while International Masters has average rating of 2411. 

Note that rating is a great predictor of who will win a chess match.

It is extremely unlikely that a chess player in the category "Other" beats a Grandmaster.

The rating can be understood as a proxy for  rationality and backward induction.

# Round 2 decimals
pd.set_option('precision', 2)

import numpy as np
title_rating1.groupby('Title1').agg(np.mean)

Let's restrict our analysis to only Grandmasters.

grandmasters = chess[chess['Title1'] == "GM"]
grandmasters

All the 26 Grandmasters stopped the Centipede game in the first round. They are the Homo Economicus described in the standard economics textbooks. 

Unfortunately, in a population of 7.8 billion people, we have only 1721 Grandmasters according with the International Chess Federation (FIDE). 

[List of Grandmasters accessed on 7/13/2020](https://ratings.fide.com/advaction.phtml?idcode=&name=&title=g&other_title=&country=&sex=&srating=0&erating=3000&birthday=&radio=rating&ex_rated=&line=desc&inactiv=&offset=1700)

grandmasters.groupby('EndNode').size()

Let's check the International Masters (IM).

international_master = chess[chess['Title1'] == "IM"]

# Return only 4 observations
international_master.head(4)

Not all International Masters stopped in the first round. 

5 International Masters stopped in the second round, and 2 in the third round.

node_IM = international_master.groupby('EndNode').size()
node_IM

These 5 International Masters that stopped in the second round represents 17% of the total numbers of International Masters.

Only 76% of International Masters acted as predicted by the neoclassical economic theory.


length_IM = len(international_master)
prop_IM = node_IM / length_IM
prop_IM

Let’s apply the same procedures to the Federation Masters. 

The proportion of Federation Masters that stopped at each node is like the International Masters. 
 

federation_master = chess[chess['Title1'] == "FM"]

node_FM = federation_master.groupby('EndNode').size()
length_FM = len(federation_master)

prop_FM = node_FM / length_FM
prop_FM

Let's put the previous descriptive statistics in a bar chart.

It will be easier to visualize the proportion of chess players at each node that ended the Centipedes game.

The bar chart suggests that Grandmasters plays Centipedes game differently from International Masters and Federation Masters. However, it looks that International Masters and Federation Masters play centipede game in a similar way. 

import plotly.graph_objects as go
node = ['Node 1', 'Node 2', 'Node 3']

fig = go.Figure(data=[
    go.Bar(name='Grandmasters', x=node, y=[1,0,0]),
    go.Bar(name='International Masters', x=node, y=prop_IM),
    go.Bar(name='Federation Masters', x=node, y=prop_FM) ])

fig.update_layout(barmode='group',
  title_text = 'Share of Chess Players that Ended Centipede Game at Each Node')

fig.show()

Let's be formal and test if the proportion of Grandmasters ($p_{g}$) is equal to the proportion of International Masters ($p_{i}$). The null hypothesis ($H_0$) is:

$$H_0: p_{g} = p_{i}$$

The z-statistic is:

$$z=\frac{\hat{p}_{g}-\hat{p}_{i}}{se(\hat{p}_{g}-\hat{p}_{i})}$$

where $se(\hat{p}_{g}-\hat{p}_{i})$ is the standard error of the difference between the sample proportions:

$$se(\hat{p}_{g}-\hat{p}_{i})=\sqrt{\frac{\hat{p}_{g}(1-\hat{p}_{g})}{n_g}+\frac{\hat{p}_{i}(1-\hat{p}_{i})}{n_i}}$$

where $n_g$ is the sample size of Grandmasters and $n_i$ is the sample size of International Masters.

For node 1, we know that:

$$\hat{p}_{g}=\frac{26}{26}=1$$

$$\hat{p}_{i}=\frac{22}{29}=0.73$$

Then:

$$z=2.68$$

The p-value of the z-statistic is 0.007.
Therefore, the null hypothesis ($H_0$) is rejected at level of significance at $\alpha=1\%$.

from statsmodels.stats.proportion import proportions_ztest

#  I inserted manually the data from Grandmasters to 
# ilustrate the input format
count = np.array([ 26, node_IM[1] ]) # number of stops
nobs = np.array([ 26, length_IM ])   # sample size

proportions_ztest(count, nobs)

Let's also test at node 1, if the proportion of International Masters ($p_{i}$) is equal to the proportion of Federation Masters ($p_{f}$). The null hypothesis ($H_0$) is:

$$H_0: p_{i} = p_{f}$$

The z-statistic is 0.18 and the respective p-value is 0.85. Therefore, we cannot reject the null hypothesis that the proportion of International Masters is equal to the proportion of Federation Masters.


count = np.array([ node_IM[1], node_FM[1] ])
nobs = np.array([ length_IM, length_FM ])

proportions_ztest(count, nobs)

## Exercises

1| Use the spreadsheet "Data Chess Tournaments" from Palacios-Huerta & Volij (2009) to calculate the proportion of other chess players that stopped in node 1, node 2, node 3, node 4, and node 5. The proportions will sum up to 100%. Other chess players is a category that excludes all chess players with the titles: Grandmasters, International Masters, and Federation Masters. 

2| This question refers to the spreadsheet "Data UPV Students-One shot" from Palacios-Huerta & Volij (2009).

a) Open the spreadsheet "Data UPV Students-One shot" here in Google Colab.  

b) How many pairs of centipede games did the students of University of Pais Vasco (UPV) play? 

c) Calculate the proportion of students that stopped in node 1, node 2, node 3, node 4, node 5, and node 6. The proportions will sum up to 100%. 

3| Compared the results of exercise 1 (Chess Players) vs exercise 2 (c) (Students). Why these two subpopulations play the centipede game differently? Speculate and justify your reasoning.

4| Use the spreadsheet "Data Chess Tournaments" from Palacios-Huerta & Volij (2009) to test if the proportion of International Masters that stopped at node 3 is equal to the proportion of other chess players that stopped at node 3 of the centipede game.

5| For this question use the spreadsheet "Data Chess Tournaments" from Palacios-Huerta & Volij (2009). Create a bar chart to compare the proportion of International Masters and other chess players that stopped the centipede game at each node. 

6| Suppose you are a neoclassical economist. How you can justify the standard economic theory built under the assumption of stronger rationality and  self-interested against the empirical evidence presented in the paper of Palacios-Huerta & Volij (2009)? Give details about your justifications.

## Reference

Fey, Mark, Richard D. McKelvey, and Thomas R. Palfrey. (1996). An Experimental Study of Constant-Sum
Centipede Games. International Journal of Game Theory, 25(3): 269–87.

Palacios-Huerta, Ignacio, and Oscar Volij. (2009). Field Centipedes. American Economic Review, 99 (4): 1619-35. [Click to download the paper](https://github.com/causal-methods/Papers/raw/master/Centipedes/Field%20Centipedes.pdf)
