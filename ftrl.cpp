// CopyRight None

#include "./ftrl.h"

ftrlOptimizer::ftrlOptimizer(float a, float b,
                             float l1, float l2) :
    alpha(a), beta(b), lambda1(l1), lambda2(l2) {
}


bool ftrlOptimizer::ftrlProcess(float* q, float* z,
                               float* W, const float& g) {

  float sigma = (sqrt(*q + g * g) - sqrt(*q)) / alpha;
  *q = *q + g * g;
  *z = *z + g - (*W) * sigma;

  bool result = true;
  
  if (std::abs(*z) < lambda1) {
    *W = 0.0f;
    result = false;
  } else {
    *W = -1.0f / (lambda2 + (beta + sqrt(*q)) / alpha) *
        (*z - lambda1 * (*z > 0.0f ? 1.0f : -1.0f));
  }
  
  return result;
}
