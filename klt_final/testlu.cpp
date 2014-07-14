#include <iostream>

#include "lu.h"

using namespace std;

int main()
{
	cout<<"Test for LU decomposition and backward substitution"<<endl;
	
	float M[9][9];
	float rhs[9];
	int i;
	
	for(i = 1 ; i < 9 ; i++) {
		for(int j = 1 ; j < 9 ; j++) {
			if(i == j)
				M[i][j] = i;
			else
				M[i][j] = 0;
		}
		rhs[i] = 1;
	}
	
	float d;
	int indx[9];
	
	ludcmp(M, 8, indx, &d);
	lubksb(M, 8, indx, rhs);
	
	for(i = 1 ; i < 9 ; i++) {
		cout<<rhs[i]<<endl;
	}
	
	return 0;
}
