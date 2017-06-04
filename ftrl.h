// CopyRight None

#ifndef FTRL_H
#define FTRL_H

#include <cmath>

class ftrlOptimizer {
 public:
  ftrlOptimizer(float a = 0.1f, float b = 1.0f,
                float l1 = 0.6f, float l2 = 1.0f);
  
  bool ftrlProcess(float* q, float* z,
                  float* W, const float& g);

 private:
  float alpha;
  float beta;
  float lambda1;
  float lambda2;

};

#endif  // FTRL_H
