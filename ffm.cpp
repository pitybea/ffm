// CopyRight None

#include <assert.h>
#include <algorithm>

#include "./ffm.h"

float safeLog(float x) {
  x = x > 1e-19f ? x : 1e-19f;
  return log(x);
}
float safeExp(float x) {
  x = -19.0f > x ? -19.0f : x;
  x = 19.0f < x ? 19.0f : x;
  return exp(x);
}


FFM::FFM(int class_n, int dim, int rank, int field, float n_theta) :
    class_num(class_n), dimension(dim), rank_num(rank), field_num(field),
    parameters(vector<ffmPara<float>>(class_n - 1,
                                      ffmPara<float>(dim, field, rank, 0.0f))),
    ftrl_q(vector<ffmPara<float>>(class_n - 1,
                                  ffmPara<float>(dim, field, rank, 0.0f))),
    ftrl_z(vector<ffmPara<float>>(class_n - 1,
                                  ffmPara<float>(dim, field, rank, 0.0f))),
    effective(vector<ffmPara<bool>>(class_n - 1,
                                    ffmPara<bool>(dim, field, rank, false))) {
  
  std::random_device rd;
  std::mt19937 generator(rd());
  std::normal_distribution<float> dist(0.0f, n_theta);
  
  for (int i = 0; i < class_num - 1; i++) {
    for (int j = 0; j < dimension; j++) {
      for (int f = 0; f < field; f++) {
        for (int k = 0; k < rank; k++) {
          parameters[i].v[j][f][k] = dist(generator);
        }
      }
    }
  }

}

float FFM::ffmFunc(const vector<int>& dims,
                   const vector<float>& vals,
                   const vector<int>& fields,
                   int class_indx,
                   vector<vector<vector<float>>>* sumCache) {

  assert(dims.size() == vals.size() && vals.size() == fields.size());
  auto result = parameters[class_indx].w0;
  for (size_t i = 0; i < dims.size(); i++) {
    auto fea_dim = dims[i];
    auto fea_val = vals[i];
    result += parameters[class_indx].w[fea_dim] * fea_val;
  }
  for (size_t i = 0; i < dims.size(); i++) {
    auto fea_dim1 = dims[i];
    auto fea_val1 = vals[i];
    auto field1 = fields[i];
    
    for (size_t j = 0; j < dims.size(); j++) {
      if (i != j) {
        auto fea_dim2 = dims[j];
        auto fea_val2 = vals[j];
        auto field2 = fields[j];
        for (int k = 0; k < rank_num; k++) {
          auto tm1 = parameters[class_indx].v[fea_dim1][field2][k] * fea_val1;
          auto tm2 = parameters[class_indx].v[fea_dim2][field1][k] * fea_val2;
          result += 0.5f * tm1 * tm2;
          if (sumCache != nullptr) {
            (*sumCache)[i][field2][k] += tm2;
            (*sumCache)[j][field1][k] += tm1;
          }
        }
      }
    }
  }
  return result;
}

vector<float> FFM::predict(const vector<int>& dims,
                           const vector<float>& vals,
                           const vector<int>& fields) {
  vector<float> exps(class_num, 1.0f);
  auto sum = 1.0f;
  for (int i = 0; i < class_num - 1; i++) {
    auto m = ffmFunc(dims, vals, fields, i);
    exps[i] = safeExp(-1.0f * m);
    sum += exps[i];
  }
  for (int i = 0; i < class_num; i++) {
    exps[i] /= sum;
  }
  return exps;
}

float FFM::loss(const vector<int>& dims,
                const vector<float>& vals,
                const vector<int>& fields,
                int class_indx) {
  auto pred = predict(dims, vals, fields);
  return -safeLog(pred[class_indx]);
}

void FFM::paraUpdate(const vector<int>& dims,
                     const vector<float>& vals,
                     const vector<int>& fields,
                     int class_indx,
                     float weight,
                     ftrlOptimizer* optimizer) {

  assert(dims.size() == vals.size() && vals.size() == fields.size());
  vector<float> exps(class_num, 1.0f);
  auto sum = 1.0f;
  vector<vector<vector<vector<float>>>> sum_cache
      (class_num -1,
       vector<vector<vector<float>>>
       (dims.size(),
        vector<vector<float>>
        (field_num,
         vector<float>(rank_num, 0.0f))));
  for (int i = 0; i < class_num - 1; i++) {
    auto m = ffmFunc(dims, vals, fields, i, &sum_cache[i]);
    exps[i] = safeExp(-1.0 * m);
    sum += exps[i];
  }
  for (int i = 0; i < class_num - 1; i++) {
    auto coef = (class_indx == i ? 1.0f : 0.0f);
    auto dl_dw0 = (coef - exps[i] / sum) * weight;
    effective[i].w0 =
        optimizer->ftrlProcess(&ftrl_q[i].w0, &ftrl_z[i].w0,
                               &parameters[i].w0, dl_dw0);

    for (size_t j = 0; j < dims.size(); j++) {
      auto fea_dim = dims[j];
      auto fea_val = vals[j];
      auto field = fields[j];
      auto dl_dw = dl_dw0 * fea_val;
      effective[i].w[fea_dim] =
          optimizer->ftrlProcess(&ftrl_q[i].w[fea_dim],
                                 &ftrl_z[i].w[fea_dim],
                                 &parameters[i].w[fea_dim],
                                 dl_dw);
      for (int k = 0; k < field_num; k++) {
        for (int m = 0; m < rank_num; m++) {
          auto dl_dv = dl_dw0 * sum_cache[i][j][k][m] * fea_val;
          effective[i].v[fea_dim][k][m] =
              optimizer->ftrlProcess(&ftrl_q[i].v[fea_dim][k][m],
                                    &ftrl_z[i].v[fea_dim][k][m],
                                    &parameters[i].v[fea_dim][k][m],
                                    dl_dv);
        }
      }
    }
  }
}


