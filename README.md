# MetaLearnBO
This is the repository for Regret bounds for the experiments in
 meta Bayesian optimization with an unknown Gaussian process prior.
 The purpose is to show the reproducibility of 
 the experimental results in the paper.
 
 We have three domains:
 1. Choosing an arm and a grasp for picking an object, with a fixed robot base pose
 2. Choosing a robot base pose and grasp for picking an object
 3. Synthetic continuous domain
 
 To reproduce any of the results, in the MetaLearnBO folder, run
```
python run_experiments -domain [ag,gbp,synth] -bo [gpucb,pi] -algorithm [zbk,commonrs,rand,plain] 
```

where -domain option specifies the domain: ag refers to the arm-and-grasp domain, gpb refers to 
the grasp-base-pose domain, and synth refers to the continuous
synthetic domain. -bo option specifies which Bayesian optimization acqusition to use:
gpucb refers to Gaussian Process Upper Confidence Bounds, and pi refers to probabilistic improvement.
-algorithm option specifies which prior estimation algorithm to use: zbk refers to our algorithm,
PEM-BO, commonrs refers to the common response surface method, which we refer to a s
TLSM-BO in our paper, rand refers to uniform random strategy, and plain refers to no prior
estimation.

## Citation
Please cite our work if you would like to use the code.
```
@inproceedings{wangkimNIPS2018,
    author={Zi Wang and Beomjoon Kim and Leslie Pack Kaelbling},
    title={Regret bounds for meta Bayesian optimization with an unknown Gaussian process prior},
    booktitle={Neural Information Processing Systems (NeurlIPS)},
    year={2018},
    url={http://people.csail.mit.edu/beomjoon/publications/zi-kim-nips18.pdf}
}
```

## References
* Regret bounds for meta Bayesian optimization with 
an unknown Gaussian process prior 
(Zi Wang*, Beomjoon Kim*, and Leslie Pack Kaelbling), 
In Neural Information Processing Systems (NeurIPS), 2018.
* Regret bounds for meta Bayesian optimization with 
an unknown Gaussian process prior 
(Zi Wang*, Beomjoon Kim*, and Leslie Pack Kaelbling), 
[arXiv](https://arxiv.org/pdf/1811.09558.pdf). 
