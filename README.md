# SciReports-FilterBubble

Repo to acompany the Scientific Reports publication: "Reframing the filter bubble through diverse scale effects in online music consumption".

Link to publication preprint: todo :) 



## Dataset Availability
* Download dataset from - todo :)
* Unzip `2023_release.zip` into `./divAtscale/data/`


## Running the Code
* To reproduce the experiements from our paper please first clone in this project, `cd` into that project dir and install packages via: `pip install -r requirements.txt`
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
If you make use of this project please cite the follow article: todo:

