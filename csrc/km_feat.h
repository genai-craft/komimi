#ifndef KOMIMI_FEAT_H
#define KOMIMI_FEAT_H
#include "km.h"
int km_num_frames(const km_model *m, int n);
void km_logmel(const km_model *m, const float *pcm, int n, float *feat, float *work);
#endif
