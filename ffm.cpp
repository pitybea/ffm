// CopyRight None

#include "./ffm.h"

#include <algorithm>

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
              int class_indx) {
  float result = parameters[class_indx].w0;
  
}

