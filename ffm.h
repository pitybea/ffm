// CopyRight None

#ifndef FFM_H
#define FFM_H

#include <vector>

#include "./ftrl.h"

using std::vector;

template<class T>
struct ffmPara {
  T w0;
  vector<T> w;
  vector<vector<vector<T>>> v;
  ffmPara(int dim, int field, int rank, T val) :
  w0(val),
    w(vector<T>(dim, val)),
    v(vector<vector<vector<T>>>
     (dim, vector<vector<T>>(field, vector<T>(rank, val)))) {
  }
};

class FFM {
 public:
  FFM(int class_n, int dim, int rank, int field, float n_theta = 1.0f / 200);
  vector<float> predict(const vector<int>& dims,
                        const vector<float>& vals,
                        const vector<int>& fields);
  void paraUpdate(const vector<int>& dims,
                  const vector<float>& vals,
                  const vector<int>& fields,
                  int class_indx,
                  float weight,
                  ftrlOptimizer* optimizer);
  float loss(const vector<int>& dims,
             const vector<float>& vals,
             const vector<int>& fields,
             int class_indx);
  
 private:
  int class_num;
  int dimension;
  int rank_num;
  int field_num;
  vector<ffmPara<float>> parameters;
  vector<ffmPara<float>> ftrl_q;
  vector<ffmPara<float>> ftrl_z;
  vector<ffmPara<bool>> effective;
  float ffmFunc(const vector<int>& dims,
                const vector<float>& vals,
                const vector<int>& fields,
                int class_indx,
                vector<vector<vector<float>>>* sumCache = nullptr);
  
      
};

#endif  // FFM_H
