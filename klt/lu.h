#ifndef LU_H
#define LU_H

void ludcmp(float [9][9], int, int [], float *);
void lubksb(float [9][9], int, int [], float []);

#endif
