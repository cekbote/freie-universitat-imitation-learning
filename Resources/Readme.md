## Literature Survey Regarding Reinforcement Learning

- [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) - Schulman et. al. (2017)
    - [Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf) - Kakade & Langford (2002)
    - [Approximately optimal approximate RL, TRPO](https://ieor8100.github.io/rl/docs/Lecture%207%20-Approximate%20RL.pdf)
    - [Approximately Optimal Approximate Reinforcement Learning (Kakade & Langford, 2002) - Blog by Konpat](https://blog.konpat.me/academic/2019/03/09/kakade-2002.html)
    - [An Introduction to the Conjugate Gradient Method Without the Agonizing Pain](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
    - [CS885 Lecture 15a: Trust Region Policy Optimization (Presenter: Shivam Kalra)](https://www.youtube.com/watch?v=jcF-HaBz0Vw&list=PLB79uOaPEEU6uU1-Pfaqr08RTTzhyB8hu&index=4)
    - [Multivariable chain rule, simple version](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/differentiating-vector-valued-functions/a/multivariable-chain-rule-simple-version)
    - [Derivatives with respect to vectors](https://www.cse.huji.ac.il/~csip/tirgul3_derivatives.pdf)
    - [Matrix Calculus](https://en.wikipedia.org/wiki/Matrix_calculus)
    - [Differentiation with respect to a vector](https://onlinelibrary.wiley.com/doi/pdf/10.1002/0471705195.app3)
    - [Conjugate Direction Methods](http://www.princeton.edu/~aaa/Public/Teaching/ORF363_COS323/F16/ORF363_COS323_F16_Lec10.pdf)
- [Recurrent World Models Facilitate Policy Evolution](https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution.pdf) - Ha & Schmidhuber (2018)
    - [World Models Paper Blog](https://worldmodels.github.io/)
    - [Recurrent World Models Facilitate Policy Evolution Talk](https://www.youtube.com/watch?v=HzA8LRqhujk)
    - [World Models - Yannic Kilcher](https://www.youtube.com/watch?v=dPsXxLyqpfs)
    - [Gaussian Mixture Models for Clustering](https://www.youtube.com/watch?v=DODphRRL79c)
    - [(ML 16.6) Gaussian mixture model (Mixture of Gaussians)](https://www.youtube.com/watch?v=Rkl30Fr2S38)
    - [Gaussian Mixture Model](https://brilliant.org/wiki/gaussian-mixture-model/)
    - [Gaussian Mixture Models Explained](https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95)

## Literature Survey Regarding Inverse Reinforcement Learning

Other Papers can be found [here.](https://github.com/dit7ya/awesome-irl)

- [Algorithms for Inverse Reinforcement Learning](http://ai.stanford.edu/~ang/papers/icml00-irl.pdf) - Andrew Ng & Stuart Russell (2000)
    - Summary: The paper talks about making sure that the expected reward (across time steps) is much greater than any other action taken. It's modelled as an optimization problem. 
- [Apprenticeship Learning via Inverse Reinforcement Learning](http://ai.stanford.edu/%7Eang/papers/icml04-apprentice.pdf) - Pieter Abbeel & Andrew Ng (2004)
    - Summary: The paper talks about matching the performace of the non-expert to the expert, by making sure that the non-expert takes actions that the expected reward matches the expected reward of the expert. It's modelled as an optimization problem.  
- [Information Theory and Statistical Mechanics](https://bayes.wustl.edu/etj/articles/theory.1.pdf) - Jaynes (1957)
    - Summary: The paper talks about the the principle of max entropy, an intuition for it and some modelling schemes based on it. 
- [Maximum Entropy Inverse Reinforcement Learning](https://new.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf) - Ziebart et. al. (2008)
    - [Lectures on Max Entropy](https://www.youtube.com/playlist?list=PLF0b3ThojznT3olRuplp5x41wUp_LZxHL)
    - [Tutorial 1](https://youtu.be/YFrtqFMglZw)
    - [Tutorial 2](https://youtu.be/4KIezIhZJ8w)
    - [Slides by Dr. Katerina Fragkiadaki](http://www.andrew.cmu.edu/course/10-703/slides/Lecture_IRL_GAIL.pdf)
        - The paper talks about using the principle of maximum entropy given certain constraints. This makes sense as the expert has demonstrations that are limited in number. Morever, you can have multiple reward functions that satisfy the actions of the expert. Similarly, you can have multiple trajectories that would satisfy the supposed goal. Hence, the principle of max entropy is used to model situations where we would make no bias about other samples / trajectories given the limited information (the expert actions). This leads to a condition where p(trajectory) is directly proportional to e ^ reward(trajectory). This makes sense too as if some trajectory has a higher probability, its reward should be higher (as the expert is choosing it). A qoute from the paper: 'Under the constraintof matching the reward value of demonstrated behavior, we employ the principle ofmaximum entropy to resolve the ambiguity in choosing a distribution over decisions.'
- [The Bidirectional Communication Theory - A Generalization of Information Theory](https://sci-hub.tw/10.1109/tcom.1973.1091610) - Marko et. al. (1973)
    - [An introduction to mutual information](https://www.youtube.com/watch?v=U9h1xkNELvY)
    - [Joint, Conditional, & Mutual Information & A Case Study](https://www.youtube.com/watch?v=jkgKWmCb5AE)
    - [HMM, MEMM, and CRF: A Comparative Analysis of Statistical Modeling Methods](https://medium.com/@Alibaba_Cloud/hmm-memm-and-crf-a-comparative-analysis-of-statistical-modeling-methods-49fc32a73586)
- [Causality,  Feedback  and  Directed Information](http://www.isiweb.ee.ethz.ch/archive/massey_pub/pdf/BI532.pdf) - Massey (1990)
    - [Causal and statistical dependence](https://probmods.org/chapters/dependence.html)
        - Summary: A causally depends on expression B if it is necessary to evaluate B in order to evaluate A. (More precisely, expression A depends on expression B if it is ever necessary to evaluate B in order to evaluate A.) We say that A and B are statistically dependent, if learning information about A tells us something about B, and vice versa. Statistical dependence is a symmetric relation between events referring to how information flows between them when we observe or reason about them. (If conditioning on A changes B, then conditioning on B also changes A. Why?) The fact that we need to be warned against confusing statistical and causal dependence suggests they are related, and indeed, they are. In general, if A causes B, then A and B will be statistically dependent. (One might even say the two notions are “causally related”, in the sense that causal dependencies give rise to statistical dependencies.) Correlation is not just a symmetrized version of causality. Two events may be statistically dependent even if there is no causal chain running between them, as long as they have a common cause (direct or indirect). 
    - [Feedback capacity and coding for channels with memory via directed information](https://www.youtube.com/watch?v=GM5_CQXAcRk)
- [Conservation of mutual and directed information](https://sci-hub.tw/10.1109/isit.2005.1523313) - Massey et. al. (2005)
- [On Directed Information and Gambling](https://arxiv.org/pdf/0802.1383.pdf) - Permuter et. al. (2008)
    - [Permuter's Notes](http://www.ee.bgu.ac.il/%7Emulti/lectures.html)
    - [Permuter's Slides](http://www.ee.bgu.ac.il/~haimp/trapdoor_slides.pdf)
- [Universal Estimation of Directed Information](https://arxiv.org/pdf/1201.2334.pdf) - Jiao et. al. (2013)
- [Guided Cost Learning](https://arxiv.org/pdf/1603.00448.pdf) - Chelsea Finn et. al. (2016). 
    - [Learning Neural Network Policies with Guided PolicySearch under Unknown Dynamics](https://papers.nips.cc/paper/5444-learning-neural-network-policies-with-guided-policy-search-under-unknown-dynamics.pdf)
        - [09:20 - 09:40 Talk: Sergey Levine, UC Berkeley - Learning Dynamic Manipulation Skills](https://www.youtube.com/watch?time_continue=171&v=CW1s6psByxk&feature=emb_title)
    - [Relative Entropy Inverse Reinforcement Learning](http://proceedings.mlr.press/v15/boularias11a/boularias11a.pdf)
    - [Learning Objective Functions for Manipulation](https://sci-hub.tw/10.1109/icra.2013.6630743) 
- [Modeling Interaction via the Principle of Maximum Causal Entropy](https://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf) - Ziebart et. al. (2010)
    - [Blog1 for this paper](https://medium.com/@jonathan_hui/rl-inverse-reinforcement-learning-56c739acfb5a)
    - [Blog2 for this paper](http://178.79.149.207/posts/maxent.html)
- [GAIL](https://cs.stanford.edu/~ermon/papers/imitation_nips2016_main.pdf) - Jonathan Ho & Stefano Ermon (2016)
    - [Medium blog explaining GAIL](https://medium.com/@sanketgujar95/generative-adversarial-imitation-learning-266f45634e60)
- [Slides summarizing various IRL algorithms](https://github.com/BAJUKA/InverseRL/blob/master/IRL_survey.pdf)
- [CARL: Controllable Agent with Reinforcement Learning for Quadruped Locomotion](https://inventec-ai-center.github.io/projects/CARL/index.html) - Luo , Soeseno et. al. (2020)
    - [This AI Controls Virtual Quadrupeds](https://www.youtube.com/watch?v=qwAiLBPEt_k&t=1s)

## Literature Survey Regarding Behavior Modelling

- [Collective Memory and Spatial Sorting in Animal Groups](https://sci-hub.se/10.1006/jtbi.2002.3065) - Couzin at. al. (2002)
	- Summary: The paper talks about creating a situation for fish interation by using a very basic modeling structure by incorporating things like zone of attraction, zone of repulsion etc. Experiments were done w.r.t the radii of these zones.. 
- [Effective leadership and decision-making in animal groups on the move](https://www.researchgate.net/publication/8042596_Effective_leadership_and_decision-making_in_animal_groups_on_the_move) - Couzin at. al. (2005)
	- Summary: The paper is an extension of the previous one where they take into account the influence of scouts / foragers on the entire group that can lead the group to food etc. Experiments were done w.r.t the number of scouts / explorers required to influence the majority. 
- [Blending in with the Shoal: Robotic FishSwarms for Investigating Strategies of GroupFormation in Guppies](https://www.researchgate.net/profile/David_Bierbach/publication/268485721_Blending_in_with_the_Shoal_Robotic_Fish_Swarms_for_Investigating_Strategies_of_Group_Formation_in_Guppies/links/54981b5c0cf2eeefc30f7016/Blending-in-with-the-Shoal-Robotic-Fish-Swarms-for-Investigating-Strategies-of-Group-Formation-in-Guppies.pdf) - Landgraf et. al. 
	- Summary: The paper talks about using a fish robot to perform certain actions similar to fish and experiment with the acceptance / reaction of other fish to the robotic fish.
- [RoboFish: increased acceptance of interactive robotic fish with realistic eyes and naturalmotion patterns by live Trinidadian guppies](https://sci-hub.tw/10.1088/1748-3190/11/1/015001) - Landgraf et. al. (2016)
	- Summary: The paper talks about using a fish robot to perform certain actions similar to fish and experiment with the acceptance / reaction of other fish to the robotic fish while changing a few things such as the movement, fish eyes etc. so that the authors would be able to observe what parameters do other fish use to treat other fish as conspecifics. Its a continuation of the previous paper. 
- [Using a robotic fish to investigate individual differences in social responsiveness in the guppy](https://www.biorxiv.org/content/10.1101/304501v1.full.pdf) - Bierbach, Landgraf et. al. (2018)
	- Summary: The paper is an extension of the above two papers and more experiments are done relations between boldness, activity and responsiveness. 
- [Learning recurrent representations for hierarchical behaviour modelling](https://arxiv.org/pdf/1611.00094.pdf) - Eyjolfsdottir et. al. (2017)
	- Summary: The paper talks about using a Ladder Network to understand as well as simulate the behavior of agents / organisms.
- [Imitation learning of fish and swarmbehavior with Recurrent Neural Networks](https://www.mi.fu-berlin.de/inf/groups/ag-ki/Theses/Completed-theses/Master_Diploma-theses/2019/Maxeiner/MA-Maxeiner.pdf) - Master thesis of Helmut Moritz Maxeiner (2019)
	- Summary: The thesis majoorly derives inspiration from the paper 'Learning recurrent representations for hierarchical behaviour modelling' where a similar network and approach is used to model the behaviour of fish using both simulated data (Couzin's data) as well as real interaction data from two female guppies similar to the data that was used in this paper 'Using a robotic fish to investigate individual differencesin social responsiveness in the guppy'. 

## Resources regarding Explainable AI
- [Restricting the Flow :Infomation Bottlenecks for Attribution](https://arxiv.org/pdf/2001.00396.pdf) - Sixt & Leon et. al. (2020)
- [How do Decisions Emerge across Layers in Neural Models? Interpretation with Differentiable Masking](https://arxiv.org/abs/2004.14992) - Cao et. al. (2020)
- [Explaining Recurrent Neural Network Predictions in Sentiment Analysis](https://arxiv.org/pdf/1706.07206.pdf) - Arras et. al. (2017)
    - [Layer-wise Relevance Propagation for Neural Networks with Local Renormalization Layers](https://arxiv.org/pdf/1604.00825.pdf) - Binder et. al. (2016)
    - [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](https://pdfs.semanticscholar.org/17a2/73bbd4448083b01b5a9389b3c37f5425aac0.pdf) - Bach & Binder et. al. (2015)
- [The Mythos of Model Interpretibility](https://arxiv.org/abs/1606.03490) - Lipton (2016)

## Resources Regarding Attention
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) - Vasvani et. al. (2017)
    - [Transformer | Attention Is All You Need | AISC Foundational](https://www.youtube.com/watch?v=S0KakHcj_rs)
    - [Transformer Neural Networks - EXPLAINED! (Attention is all you need)](https://www.youtube.com/watch?v=TQQlZhbC5ps)
- [Attention by Selection: A Deep Selective Attention Approach to Breast Cancer Classification](https://sci-hub.tw/10.1109/tmi.2019.2962013) - Xu et. al. (2019)
    - [Attention in Neural Networks](https://www.youtube.com/watch?v=W2rWgXJBZhU)

## Resources used for Creating a Ladder Network 
- [Blog on Ladder Networks.](https://towardsdatascience.com/a-new-kind-of-deep-neural-networks-749bcde19108)
- [Keras LSTM parameters](https://www.youtube.com/watch?v=LZ-GS7LOyWs)
- [My implementation of a prototype Ladder Network](https://colab.research.google.com/drive/1mj1n5MteLofN7BEE-y4hNVIok0RIdm-i?usp=sharing)

## Resources used for Creating a Convolutional LSTM
- [An introduction to ConvLSTM](https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7)
- [Jupyter Notebook Tips and Tricks](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)
- [Transformers and Layer Normalization](https://www.tensorflow.org/tutorials/text/transformer)

