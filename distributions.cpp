#include <math.h>
#include "neural_network.h"


//constants for computing normal distribution cdf
#define A1 0.254829592
#define A2 -0.284496736
#define A3 1.421413741
#define A4 -1.453152027
#define A5 1.061405429
#define P 0.3275911

DTYPE normal_cdf(DTYPE x){
    // Save the sign of x
    WORD sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);

    // A&S formula 7.1.26
    DTYPE t = 1.0/(1.0 + P*x);
    DTYPE y = 1.0 - (((((A5*t + A4)*t) + A3)*t + A2)*t + A1)*t*exp(-x*x);

    return 0.5*(1.0 + sign*y);
}

