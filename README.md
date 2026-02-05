# Reframing the Filter Bubble: Suporting Code

Repo to acompany the Scientific Reports publication: "Reframing the filter bubble through diverse scale effects in online music consumption"



## Dataset Availability
* Download dataset from [link](https://kaggle.com/datasets/4b70ea10861ed6cad92277d498ccaf3d621a766b3d982bdba0321b1038ac29a4)
* Unzip `2023_release.zip` into `./divAtscale/data/`


## Running the Code
* run experiments by calling `run_expr.sh`
* To modify experiments settings edit expr_config.json 

```
// expr_config.json params: 
"year" : 2023 : year of dataset (modify if working with other dataset)
"n_sample" : number of users to random sample (for testing)
"r_seed" : random seed (int)
```

## Documentation 
Complete documentation for the helper functions and BiCM code and can be found here: ([link](https://deshiiii.github.io/div-scale-documentation/divAtScale/index.html))

## Credits
This projects makes use of the BiCM as a null model. The full code by Mika J. Straka which this project makes use of can be found here:
https://github.com/tsakim/bicm
All credits for this code go to these authors. 

## Citing 
If you make use of this project please cite the follow article: <mark>todo</mark>:

