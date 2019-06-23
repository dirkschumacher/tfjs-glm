# tfjs-glm
Generalized linear models in tensorflow.js using Iteratively Re-Weighted Least Squares (WIP)

Experimental and work in progress. Use at own risk. Still numerical problems, especially with log link.

## Families

* Gaussian (links: identity)
* Binomial (links: logit)
* Poisson (links: log)

## API

```js
const {glm, gaussian, binomial, poisson} = require("tfjs-glm")

// linear regression
const coefficents = glm(designMatrix, response, gaussian("identity"))

// logistic regression
const coefficents = glm(designMatrix, response, binomial("logit"))

// poisson regression
const coefficents = glm(designMatrix, response, poisson("log"))
```

## References

1: Arnold, T., Kane, M., & Lewis, B. W. (2019). A Computational Approach to Statistical Learning. CRC Press.